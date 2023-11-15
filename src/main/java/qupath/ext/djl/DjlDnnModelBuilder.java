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

import java.net.URI;
import java.nio.file.Files;

import ai.djl.engine.Engine;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.LayoutType;
import qupath.lib.common.GeneralTools;
import qupath.opencv.dnn.DnnModel;
import qupath.opencv.dnn.DnnModelBuilder;
import qupath.opencv.dnn.DnnModelParams;

/**
 * A {@link DnnModelBuilder} using Deep Java Library.
 * 
 * @author Pete Bankhead
 */
public class DjlDnnModelBuilder implements DnnModelBuilder {

	private static String getEngineName(String framework) {
		if (DjlTools.ALL_ENGINES.contains(framework))
			return framework;
		
		switch(framework) {
		case DnnModelParams.FRAMEWORK_TENSORFLOW:
			return DjlTools.ENGINE_TENSORFLOW;
		case DnnModelParams.FRAMEWORK_TF_LITE:
			return DjlTools.ENGINE_TFLITE;
		case DnnModelParams.FRAMEWORK_ONNX_RUNTIME:
			return DjlTools.ENGINE_ONNX_RUNTIME;
		case DnnModelParams.FRAMEWORK_PYTORCH:
			return DjlTools.ENGINE_PYTORCH;
		case DnnModelParams.FRAMEWORK_MXNET:
			return DjlTools.ENGINE_MXNET;
		default:
			return null;
		}
	}
	
	
	private static String estimateEngine(URI uri) {
		var urlString = uri.toString().toLowerCase();
		if (urlString.endsWith(".onnx")) {
			if (Engine.hasEngine(DjlTools.ENGINE_ONNX_RUNTIME))
				return DjlTools.ENGINE_ONNX_RUNTIME;
		}
		
		if (urlString.endsWith("pytorch") || urlString.endsWith(".pt")) {
			if (Engine.hasEngine(DjlTools.ENGINE_PYTORCH))
				return DjlTools.ENGINE_PYTORCH;
		}

		if (urlString.endsWith(".tflite")) {
			if (Engine.hasEngine(DjlTools.ENGINE_TFLITE))
				return DjlTools.ENGINE_TFLITE;
		}
		
		var path = GeneralTools.toPath(uri);
		if (path != null) {
			if ("saved_model.pb".equals(path.getFileName().toString()) || 
					(Files.isDirectory(path) && Files.exists(path.resolve("saved_model.pb")))) {
				if (Engine.hasEngine(DjlTools.ENGINE_TENSORFLOW))
					return DjlTools.ENGINE_TENSORFLOW;
			}
		}
		return null;
	}
	
	private static LayoutType getLayout(char c) {
		switch (c) {
		case 'b': return LayoutType.BATCH;
		case 't': return LayoutType.TIME;
		case 'c': return LayoutType.CHANNEL;
		case 'z': return LayoutType.DEPTH;
		case 'y': return LayoutType.HEIGHT;
		case 'x': return LayoutType.WIDTH;
		case 'i': return LayoutType.UNKNOWN; // Batch?
		default:
			return LayoutType.UNKNOWN;
		}		
	}
	
	/**
	 * Convert a model zoo axes String to a DJL layout string
	 * @param axes
	 * @return
	 */
	private static String axesToLayout(String axes) {
		if (axes == null)
			return null;
		axes = axes.strip().toLowerCase();
		var sb = new StringBuilder(axes.length());
		for (var c : axes.toCharArray()) {
			sb.append(getLayout(c).getValue());
		}
		return sb.toString();
	}
	
	@Override
	public DnnModel buildModel(DnnModelParams params) {
		var framework = params.getFramework();
		String engineName = null;
		if (framework == null) {
			var uris = params.getURIs();
			if (uris.isEmpty())
				return null;
			engineName = estimateEngine(uris.get(0));
		} else
			engineName = getEngineName(framework);
		if (engineName == null || !Engine.hasEngine(engineName)) {
			return null;
		}
		String layout = params.getLayout();
		if (layout != null) {
			layout = axesToLayout(layout);
		}
		
		return new DjlDnnModel(
				engineName,
				params.getURIs(),
				layout,
				params.getInputs(),
				params.getOutputs(),
				params.requestLazyInitialize());
	}
	
}