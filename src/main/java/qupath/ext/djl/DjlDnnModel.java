/*-
 * Copyright 2022 QuPath developers, University of Edinburgh
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package qupath.ext.djl;

import java.io.IOException;
import java.net.URI;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

import org.bytedeco.opencv.opencv_core.Mat;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Batchifier;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;
import qupath.lib.io.UriResource;
import qupath.opencv.dnn.BlobFunction;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnShape;
import qupath.opencv.dnn.PredictionFunction;

class DjlDnnModel implements DnnModel<NDList>, AutoCloseable, UriResource {

	private static final Logger logger = LoggerFactory.getLogger(DjlDnnModel.class);

	private List<URI> uris;
	private String engine;
	private String ndLayout;

	private Map<String, DnnShape> inputs;
	private Map<String, DnnShape> outputs;
	
	private boolean lazyInitialize;

	private transient boolean failed;
	private transient ZooModel<NDList, NDList> model;
	private transient Predictor<NDList, NDList> predictor;
	private transient BlobFunction<NDList> blobFun;
	private transient PredictionFunction<NDList> predictFun;

	DjlDnnModel(String engine, Collection<URI> uris, String ndLayout, Map<String, DnnShape> inputs, Map<String, DnnShape> outputs, boolean lazyInitialize) {
		this.engine = engine;
		this.uris = new ArrayList<>(uris);
		if (ndLayout == null) {
			logger.warn("ndLayout not specified - I'll need to try to guess");
		} else
			this.ndLayout = ndLayout.toUpperCase();
		this.inputs = inputs;
		this.outputs = outputs;
		this.lazyInitialize = lazyInitialize;
		if (!lazyInitialize)
			ensureInitialized();
	}

	private void ensureInitialized() {
		if (model != null)
			return;
		if (!failed && model == null) {
			synchronized (this) {
				if (!failed && model == null) {
					try {
						logger.debug("Initializing DjlDnnModel");
						model = DjlTools.loadModel(engine, uris.toArray(URI[]::new));
						if (ndLayout != null && ndLayout.contains("N"))
							predictor = model.newPredictor();
						else
							predictor = model.newPredictor(new NoopTranslator(Batchifier.STACK));

						// TODO: Better handling of missing inputs/outputs - we may need to run a prediction for this to work
						if (this.inputs == null || this.inputs.isEmpty()) {
							var description = model.describeInput();
							if (description != null && !description.isEmpty())
								inputs = description.stream().collect(Collectors.toMap(p -> p.getKey(), p -> DjlTools.convertShape(p.getValue())));
							else
								inputs = Map.of(DnnModel.DEFAULT_INPUT_NAME, DnnShape.UNKNOWN_SHAPE);
						}

						if (this.outputs == null || this.outputs.isEmpty()) {
							var description = model.describeOutput();
							if (description != null && !description.isEmpty())
								outputs = description.stream().collect(Collectors.toMap(p -> p.getKey(), p -> DjlTools.convertShape(p.getValue())));
							else
								outputs = Map.of(DnnModel.DEFAULT_OUTPUT_NAME, DnnShape.UNKNOWN_SHAPE);
						}

						blobFun = new BlobFun();
						predictFun = new PredictFun();
					} catch (Exception e) {
						failed = true;
						logger.debug("Failed to create DjlDnnModel");
						throw new RuntimeException(e);
					}
				}
			}
		}
	}

	@Override
	public BlobFunction<NDList> getBlobFunction() {
		ensureInitialized();
		return blobFun;
	}

	@Override
	public BlobFunction<NDList> getBlobFunction(String name) {
		ensureInitialized();
		return blobFun;
	}

	@Override
	public PredictionFunction<NDList> getPredictionFunction() {
		ensureInitialized();
		return predictFun;
	}

	@Override
	public synchronized void close() throws Exception {
		if (model != null) {
			model.close();
			model = null;
			blobFun = null;
			predictFun = null;
			logger.debug("Closed DjlDnnModel");
		}
	}

	private static final String DEFAULT_MAT_LAYOUT = getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);
	
	private static String getLayout(LayoutType... layouts) {
		return LayoutType.toString(layouts);
	}
	
	private static String estimateInputLayout(Mat mat) {
		if (mat.channels() >= 1)
			return DEFAULT_MAT_LAYOUT;
		var sizes = mat.createIndexer().sizes();
		switch (sizes.length) {
		case 1: return getLayout(LayoutType.HEIGHT);
		case 2: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH);
		case 3: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);
		default:
			throw new IllegalArgumentException("Unknown layout for input Mat " + mat);
		}
	}
	
	private static String estimateOutputLayout(NDArray array) {
		var shape = array.getShape();
		if (shape.isLayoutKnown())
			return shape.toLayoutString();
		switch (shape.dimension()) {
		case 1: return getLayout(LayoutType.HEIGHT);
		case 2: return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH);
		case 3: 
			// Assume channels-first or channels-last, and the channels dimensions is shorter
			if (shape.get(2) < shape.get(0))
				return getLayout(LayoutType.HEIGHT, LayoutType.WIDTH, LayoutType.CHANNEL);
			else
				return getLayout(LayoutType.CHANNEL, LayoutType.HEIGHT, LayoutType.WIDTH);
		default:
			throw new IllegalArgumentException("Unknown layout for output shape " + shape);
		}
	}
	
	

	private class BlobFun implements BlobFunction<NDList> {

		@Override
		public NDList toBlob(Mat... mats) {
			NDList list = new NDList();
			String layout = ndLayout;
			for (var mat : mats) {
				// Try to figure out the layout
				if (layout == null) {
					layout = estimateInputLayout(mat);
				}
				list.add(DjlTools.matToNDArray(model.getNDManager(), mat, layout));
			}
			return list;
		}

		@Override
		public List<Mat> fromBlob(NDList blob) {
			String layout;
			if ((ndLayout == null || ndLayout.length() != blob.singletonOrThrow().getShape().dimension()) && !blob.isEmpty())
				layout = estimateOutputLayout(blob.get(0));
			else
				layout = ndLayout;
			var output = blob.stream().map(b -> DjlTools.ndArrayToMat(b, layout)).collect(Collectors.toList());
			blob.close();
			return output;
		}

	}

	private class PredictFun implements PredictionFunction<NDList> {

		@Override
		public NDList predict(NDList input) {
			try {
				NDList output;
				// TODO: Check whether to support per-thread predictors
				synchronized (predictor) {
					output = predictor.batchPredict(Collections.singletonList(input)).get(0);
				}
				input.close();
				return output;
			} catch (TranslateException e) {
				throw new RuntimeException(e);
			}
		}

		@Override
		public Map<String, DnnShape> getInputs() {
			return inputs;
		}

		@Override
		public Map<String, DnnShape> getOutputs(DnnShape... inputShapes) {
			return outputs;
		}

	}

	@Override
	public Collection<URI> getURIs() throws IOException {
		return new ArrayList<>(uris);
	}

	@Override
	public boolean updateURIs(Map<URI, URI> replacements) throws IOException {
		var newUris = uris.stream().map(u -> replacements.getOrDefault(u, u)).collect(Collectors.toList());
		if (Objects.equals(newUris, uris))
			return false;
		this.uris = newUris;
		return true;
	}

}