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

package qupath.ext.djl.ui;

import java.nio.file.Files;
import java.util.ResourceBundle;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

import ai.djl.util.cuda.CudaUtils;
import javafx.geometry.Pos;
import javafx.scene.Cursor;
import javafx.scene.Node;
import javafx.scene.text.TextAlignment;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.util.Utils;
import javafx.application.Platform;
import javafx.beans.binding.Bindings;
import javafx.beans.property.ObjectProperty;
import javafx.beans.property.SimpleObjectProperty;
import javafx.collections.FXCollections;
import javafx.collections.ObservableMap;
import javafx.geometry.Insets;
import javafx.geometry.Orientation;
import javafx.scene.Scene;
import javafx.scene.control.Alert.AlertType;
import javafx.scene.control.Button;
import javafx.scene.control.ButtonType;
import javafx.scene.control.ContentDisplay;
import javafx.scene.control.Label;
import javafx.scene.control.Separator;
import javafx.scene.control.Tooltip;
import javafx.scene.layout.GridPane;
import javafx.scene.shape.Circle;
import javafx.stage.Modality;
import javafx.stage.Stage;
import qupath.ext.djl.DjlTools;
import qupath.fx.dialogs.Dialogs;
import qupath.fx.utils.GridPaneUtils;
import qupath.lib.common.GeneralTools;
import qupath.lib.common.ThreadTools;
import qupath.lib.gui.QuPathGUI;
import qupath.lib.gui.tools.GuiTools;

/**
 * Command to display Deep Java Library engines.
 * 
 * @author Pete Bankhead
 */
public class DjlEngineCommand {
	
	private static final Logger logger = LoggerFactory.getLogger(DjlEngineCommand.class);
	
	private static final String TITLE = "Deep Java Library engines";
	
	private QuPathGUI qupath;

	private static final ResourceBundle bundle = ResourceBundle.getBundle("qupath.ext.djl.ui.strings");
	
	private enum EngineStatus {
		UNAVAILABLE, UNKNOWN, PENDING, AVAILABLE, FAILED
	}
	
	private static DjlEngineCommand INSTANCE;
	
	private Stage stage;
	
	/**
	 * Engine only supported on Linux
	 */
	private static Set<String> LINUX_ONLY = Set.of(
			DjlTools.ENGINE_TENSORRT
			);
	
	/**
	 * Several engines definitely don't work on Apple Silicon (for now anyway).
	 */
	private static Set<String> UNSUPPORTED_APPLE_SILICON = Set.of(
			DjlTools.ENGINE_PADDLEPADDLE,
			DjlTools.ENGINE_TENSORFLOW,
			DjlTools.ENGINE_TFLITE,
			DjlTools.ENGINE_LIGHTGBM
			);
	
	/**
	 * Maintain a map to indicate if the engine is available or not
	 */
	private ObservableMap<String, ObjectProperty<EngineStatus>> available = FXCollections.observableHashMap();
	
	DjlEngineCommand(QuPathGUI qupath) {
		this.qupath = qupath;
		init();
	}
	
	private void init() {
		var pane = new GridPane();
		pane.getStylesheets().add(DjlEngineCommand.class.getResource("styles.css").toExternalForm());
		pane.setHgap(5);
		pane.setVgap(5);
		pane.setPadding(new Insets(10));
		int row = 0;

		var label = createCenteredLabelForString(
				String.format(bundle.getString("overview.description"), Engine.getDjlVersion())
		);
		GridPaneUtils.addGridRow(pane, row++, 0, null, label, label, label);
		var label2 = createCenteredLabelForKey("overview.directories");
		GridPaneUtils.addGridRow(pane, row++, 0, null, label2, label2, label2);
		var label3 = createCenteredLabelForKey("overview.licensing");
		GridPaneUtils.addGridRow(pane, row++, 0, null, label3, label3, label3);

		var separator = new Separator(Orientation.HORIZONTAL);
		GridPaneUtils.addGridRow(pane, row++, 0, null, separator, separator, separator);

		var labelCuda = createCenteredLabelForString(getCudaString());
		GridPaneUtils.addGridRow(pane, row++, 0, null, labelCuda, labelCuda, labelCuda);

		for (var name : Engine.getAllEngines()) {
			separator = new Separator(Orientation.HORIZONTAL);
			GridPaneUtils.addGridRow(pane, row++, 0, null, separator, separator, separator);
			GridPaneUtils.setToExpandGridPaneWidth(separator);
			
			var status = available.computeIfAbsent(name, n -> new SimpleObjectProperty<>(EngineStatus.UNKNOWN));

			var path = Utils.getEngineCacheDir(name);

			String tooltip = String.format(bundle.getString("tooltip.engine"), name);
			String labelNameText = name;
			if (name.equals(Engine.getDefaultEngineName())) {
				labelNameText = String.format(bundle.getString("label.defaultName"), name);
			}
			var labelName = new Label(labelNameText);
			labelName.setStyle("-fx-font-weight: bold;");
			GridPaneUtils.addGridRow(pane, row++, 0, tooltip, labelName, labelName);

			var labelPath = new Label(path.toString());
			var btnDownload = new Button();
			btnDownload.setText(bundle.getString("button.download"));
			btnDownload.setPrefWidth(60.0);
			long timeoutMillis = 500L;
			btnDownload.setOnAction(e -> checkEngineStatus(name, status, timeoutMillis, false));
			btnDownload.disableProperty().bind(Bindings.createBooleanBinding(() -> 
				status.get() == EngineStatus.AVAILABLE || status.get() == EngineStatus.PENDING,
				status));
			btnDownload.textProperty().bind(Bindings.createStringBinding(() -> {
				switch (status.get()) {
				case AVAILABLE:
					return bundle.getString("button.download.available");
				case FAILED:
					return bundle.getString("button.download.failed");
				case PENDING:
					return bundle.getString("button.download.pending");
				case UNAVAILABLE:
					return bundle.getString("button.download.unavailable");
				case UNKNOWN:
				default:
					return bundle.getString("button.download.unknown");
				}
			}, status));
			
			var tooltipDownload = new Tooltip();
			tooltipDownload.textProperty().bind(Bindings.createStringBinding(() -> {
				switch (status.get()) {
				case AVAILABLE:
					return name + " is available";
				case UNAVAILABLE:
					return name + " is not available - click to download it";
				case FAILED:
					return name + " download & initialization failed - you can try again, but it may not be supported on this platform";
				case PENDING:
					return name + " download pending";
				case UNKNOWN:
				default:
					return name + " download status is unknown";				
				}
			}, status));
			btnDownload.setTooltip(tooltipDownload);
			
			var circle = new Circle(6.0);
			updateIconFromStatus(status.get(), circle);


			var labelPathLabel = createLabelForKey("label.path");
			labelPathLabel.setLabelFor(labelPath);
			labelPathLabel.setContentDisplay(ContentDisplay.LEFT);

			var tooltipPath = new Tooltip();
			labelPath.setTooltip(tooltipPath);
			labelPath.setCursor(Cursor.HAND);
			labelPath.setOnMouseClicked(e -> {
				if (e.getClickCount() == 2 && Files.isDirectory(path))
					GuiTools.browseDirectory(path.toFile());
				else
					logger.debug("Cannot open {} - directory does not exist", path);
			});

			var labelVersion = new Label("");
			labelVersion.setMaxWidth(Double.MAX_VALUE);

			var labelVersionLabel = createLabelForKey("label.version");
			labelVersionLabel.setLabelFor(labelVersion);
			labelVersionLabel.setContentDisplay(ContentDisplay.LEFT);

			status.addListener((v, o, n) -> {
				if (Files.isDirectory(path)) {
					tooltipPath.setText(bundle.getString("tooltip.openPath"));
				} else {
					labelPath.setStyle("-fx-opacity: 0.5;");
					tooltipPath.setText(bundle.getString("tooltip.pathMissing"));
				}
				updateIconFromStatus(n, circle);
				updateVersionFromStatus(n, name, labelVersion);
			});

			pane.add(circle, 0, row, 1, 2);

			GridPaneUtils.addGridRow(pane, row++, 1, null, labelPathLabel, labelPath);
			GridPaneUtils.addGridRow(pane, row++, 1, null, labelVersionLabel, labelVersion);
			GridPaneUtils.addGridRow(pane, row++, 1, null, btnDownload, btnDownload);
			
			GridPaneUtils.setToExpandGridPaneWidth(labelName, labelPath, btnDownload);

			// Update the engine status quietly
			checkEngineStatus(name, status, -1, true);
		}
		
		stage = new Stage();
		stage.setTitle(TITLE);
		stage.initOwner(QuPathGUI.getInstance().getStage());
		stage.initModality(Modality.APPLICATION_MODAL);
		stage.setScene(new Scene(pane));
	}

	private static void updateIconFromStatus(EngineStatus status, Node node) {
		if (status == null)
			status = EngineStatus.UNKNOWN;
		node.getStyleClass().setAll("djl-engine-status", status.name().toString().toLowerCase());
	}

	private static void updateVersionFromStatus(EngineStatus status, String engineName, Label labelVersion) {
		if (status == EngineStatus.AVAILABLE) {
			try {
				var engine = Engine.getEngine(engineName);
				labelVersion.setText(engine.getVersion());
				labelVersion.setTooltip(new Tooltip(engine.toString()));
				return;
			} catch (Exception e) {
				logger.error("Error updating engine version: {}", e.getMessage(), e);
			}
		}
		labelVersion.setText("");
		labelVersion.setTooltip(null);
	}


	private static Label createLabelForKey(String key) {
		var label = new Label(bundle.getString(key));
		label.setWrapText(true);
		return label;
	}


	private static Label createCenteredLabelForKey(String key) {
		return createCenteredLabelForString(bundle.getString(key));
	}

	private static Label createCenteredLabelForString(String s) {
		var label = new Label(s);
		label.setWrapText(true);
		label.setAlignment(Pos.CENTER);
		label.setTextAlignment(TextAlignment.CENTER);
		label.setMaxHeight(Double.MAX_VALUE);
		label.setMaxWidth(Double.MAX_VALUE);
		return label;
	}


	private void showDialog() {
		if (stage.isShowing())
			stage.toFront();
		else
			stage.show();
	}

	/**
	 * Get a user-friendly string describing the CUDA version and GPUs.
	 * @return
	 */
	private static String getCudaString() {
		if (CudaUtils.hasCuda()) {
			var sb = new StringBuilder()
					.append(bundle.getString("label.cudaVersion"))
					.append(CudaUtils.getCudaVersionString());
			for (int i = 0; i < CudaUtils.getGpuCount(); i++) {
				sb.append("\n")
						.append("  GPU ")
						.append(i)
						.append(" ")
						.append(bundle.getString("label.computeCapability"))
						.append(CudaUtils.getComputeCapability(i));
			}
			return sb.toString();
		} else if (GeneralTools.isMac() && "aarch64".equals(System.getProperty("os.arch"))) {
			return bundle.getString("cuda.appleSilicon");
		} else {
			return bundle.getString("cuda.notFound");
		}
	}


	private void checkEngineStatus(String name, ObjectProperty<EngineStatus> status, long timeoutMillis, boolean doQuietly) {
		// Request the engine in a background thread, triggering download if necessary
		var pool = Executors.newSingleThreadExecutor(ThreadTools.createThreadFactory("djl-engine-request", true));
		updateStatus(status, EngineStatus.PENDING);
		var future = pool.submit((Callable<Boolean>)() -> checkEngineAvailability(name, status, doQuietly));
		if (timeoutMillis <= 0)
			return;
		pool.shutdown();
		try {
			// Wait until the timeout - engine might already be available & return quickly
			var result = future.get(timeoutMillis, TimeUnit.MILLISECONDS);
			if (result != null && result.booleanValue()) {
				Dialogs.showInfoNotification(TITLE, name + " is available!");
				return;
			} else
				logger.debug("No engine available for {}", name);
		} catch (InterruptedException e) {
			logger.error("Requesting for engine " + name + "interrupted!", e);
		} catch (ExecutionException e) {
			logger.error("Error requesting engine " + name + ": " + e.getLocalizedMessage(), e);
		} catch (TimeoutException e) {
			logger.debug("Request for engine {} timed out - likely either initializing or downloading", name);
		}
	}
	
	private void updateStatus(ObjectProperty<EngineStatus> status, EngineStatus newStatus) {
		if (Platform.isFxApplicationThread()) {
			status.set(newStatus);
		} else {
			Platform.runLater(() -> updateStatus(status, newStatus));
		}
	}
	
	
	private Boolean checkEngineAvailability(String name, ObjectProperty<EngineStatus> status, boolean doQuietly) {

		if (!GeneralTools.isLinux() && LINUX_ONLY.contains(name)) {
			if (!doQuietly)
				Dialogs.showErrorMessage(TITLE, name + " is only available on Linux, sorry");
			updateStatus(status, EngineStatus.UNAVAILABLE);
			return Boolean.FALSE;
		}

		// For Apple Silicon, we may not know if we are running under Rosetta or not
		// TODO: Check for custom aarch64 build
		if (!doQuietly && GeneralTools.isMac() && UNSUPPORTED_APPLE_SILICON.contains(name)) {
			var button = Dialogs.builder()
				.title(TITLE)
				.alertType(AlertType.WARNING)
				.contentText(name + " won't work on recent Macs with Apple Silicon.\n"
						+ "Do you want to continue anyway?")
				.buttons(ButtonType.YES, ButtonType.NO)
				.showAndWait()
				.orElse(ButtonType.NO);
			if (ButtonType.NO.equals(button)) {
				updateStatus(status, EngineStatus.UNAVAILABLE);
				return Boolean.FALSE;
			}
		}
		
		try {
			// Allow engine downloads if not doQuietly
			updateStatus(status, EngineStatus.PENDING);
			boolean isAvailable;
			if (doQuietly)
				isAvailable = DjlTools.isEngineAvailable(name);
			else
				isAvailable = DjlTools.getEngine(name, true) != null;
			if (isAvailable) {
				updateStatus(status, EngineStatus.AVAILABLE);
				if (!doQuietly)
					Dialogs.showInfoNotification(TITLE, name + " is now available!");
			} else {
				updateStatus(status, EngineStatus.UNAVAILABLE);
				if (!doQuietly) {
					Dialogs.showWarningNotification(TITLE, "Unable to initialize " + name + ", sorry");
				}
			}
			return isAvailable;
		} catch (EngineException e) {
			if (doQuietly) {
				logger.debug("Error requesting engine: " + e.getLocalizedMessage(), e);				
			} else {
				logger.error("Error requesting engine: " + e.getLocalizedMessage(), e);
				Dialogs.showErrorMessage(TITLE, "Unable to initialize " + name + "\n- engine might not be supported on this platform, sorry");
			}
		} catch (Exception e) {
			if (doQuietly) {
				logger.debug("Error requesting engine: " + e.getLocalizedMessage(), e);				
			} else {
				logger.error("Error requesting engine: " + e.getLocalizedMessage(), e);
				Dialogs.showErrorMessage(TITLE, "Exception when trying to download " + name + "\n" + e.getLocalizedMessage());
			}
		}
		updateStatus(status, EngineStatus.FAILED);
		return Boolean.FALSE;
	}
	
	/**
	 * Show dialog.
	 * @param qupath
	 */
	public static void showDialog(QuPathGUI qupath) {
		if (INSTANCE == null)
			INSTANCE = new DjlEngineCommand(qupath);
		INSTANCE.showDialog();
	}
	

}
