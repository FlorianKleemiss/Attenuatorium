import sys
import os
import numpy as np


# Function to check and install missing dependencies
def check_and_install_dependencies():
    required_packages = ['PyQt6', 'numpy', 'matplotlib', 'scipy']
    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        import subprocess
        for package in missing_packages:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print("All required packages installed successfully.")

# Check and install dependencies
check_and_install_dependencies()

# Import PyQt6 components
try:
    # Import QtWidgets classes individually to satisfy linter
    from PyQt6 import QtWidgets  # noqa: E402
    QApplication = QtWidgets.QApplication
    QMainWindow = QtWidgets.QMainWindow
    QVBoxLayout = QtWidgets.QVBoxLayout
    QHBoxLayout = QtWidgets.QHBoxLayout
    QLabel = QtWidgets.QLabel
    QPushButton = QtWidgets.QPushButton
    QSlider = QtWidgets.QSlider
    QFileDialog = QtWidgets.QFileDialog
    QWidget = QtWidgets.QWidget
    QGroupBox = QtWidgets.QGroupBox
    QSplitter = QtWidgets.QSplitter
    QComboBox = QtWidgets.QComboBox
    QProgressBar = QtWidgets.QProgressBar
    
    # Import QtCore classes
    from PyQt6 import QtCore  # noqa: E402
    Qt = QtCore.Qt
    QCoreApplication = QtCore.QCoreApplication
except ImportError as e:
    print(f"Error importing PyQt6 components: {e}")
    sys.exit(1)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg # noqa: E402
from matplotlib.figure import Figure # noqa: E402

class MatplotlibCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, width=12, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_log = self.fig.add_subplot(121)
        self.ax_linear = self.fig.add_subplot(122)
        super(MatplotlibCanvas, self).__init__(self.fig)

# Import the hkl functionality
from attenuatorium import read_hkl, reflection_list # noqa: E402

class AttenuatoriumGUI(QMainWindow):
    COLOR_USE_STRONG = 'gray'      # Use Strong data (lowest intensities)
    COLOR_USE_BOTH = 'dodgerblue'  # Use both data (middle region)
    COLOR_USE_WEAK = 'crimson'     # Use weak data (highest intensities)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attenuatorium GUI")
        self.setGeometry(100, 100, 1600, 900) # Adjusted default size

        self.reflection_sets = []
        self.file_names = []
        self.merged_data = None
        self.common_i_strong_np = np.array([]) # For live threshold updates

        # Threshold factors (percentiles)
        self.low_threshold_factor = 0.25  # Default to 25th percentile
        self.high_threshold_factor = 0.75 # Default to 75th percentile
        self.weak_low_threshold_factor = 0.25
        self.weak_high_threshold_factor = 0.75

        # Store actual threshold values
        self.strong_low_threshold_val = 0
        self.strong_high_threshold_val = np.inf
        self.weak_low_threshold_val = 0
        self.weak_high_threshold_val = np.inf
        self.low_thresh_line = None
        self.high_thresh_line = None

        # Matplotlib plot canvas
        self.canvas = MatplotlibCanvas(self, width=12, height=6, dpi=100)

        # Attributes for plot elements for live updates
        self.low_thresh_line_log = None
        self.high_thresh_line_log = None
        self.weak_low_thresh_line_log = None
        self.weak_high_thresh_line_log = None

        self.s_low_line_lin = None
        self.s_high_line_lin = None
        self.w_low_line_lin = None
        self.w_high_line_lin = None

        self.sorted_strong_intensities = None
        self.sorted_weak_intensities = None
        self.linear_regression_slope_origin = None

        self._init_ui()

    def _init_ui(self):
        """Initialize the user interface components"""
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Create a splitter to divide the UI
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Top section: Controls
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)

        # File selection controls
        file_group = QGroupBox("File Selection")
        file_layout = QHBoxLayout()

        self.load_btn = QPushButton("Load HKL Files")
        self.load_btn.clicked.connect(self.load_files)

        self.file_combo1 = QComboBox() # Strong file
        self.file_combo1.currentIndexChanged.connect(self.update_plots)

        self.file_combo2 = QComboBox() # Weak file
        self.file_combo2.currentIndexChanged.connect(self.update_plots)

        # Swap button
        self.swap_btn = QPushButton("⇄")
        self.swap_btn.setToolTip("Swap Strong/Weak files")
        self.swap_btn.setFixedWidth(28)
        self.swap_btn.clicked.connect(self.swap_files)

        file_layout.addWidget(self.load_btn)
        file_layout.addWidget(QLabel("Strong File:"))
        file_layout.addWidget(self.file_combo1)
        file_layout.addWidget(self.swap_btn)
        file_layout.addWidget(QLabel("Weak File:"))
        file_layout.addWidget(self.file_combo2)
        file_group.setLayout(file_layout)
        threshold_group = QGroupBox("Statistics")
        threshold_layout = QVBoxLayout()

        # Threshold controls for both Strong and Weak Set
        weak_threshold_group = QGroupBox("Threshold Controls")
        weak_threshold_layout = QVBoxLayout()
        weak_low_layout = QHBoxLayout()
        weak_low_layout.addWidget(QLabel("Low Cutoff Percentile:"))
        self.weak_low_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.weak_low_threshold_slider.setRange(0, 100)
        self.weak_low_threshold_slider.setValue(int(self.weak_low_threshold_factor * 100))
        self.weak_low_threshold_slider.valueChanged.connect(lambda value, name="weak_low": self._update_threshold_sliders(name, value))
        self.weak_low_threshold_slider.sliderReleased.connect(self.update_plots) # Moved here

        self.weak_low_label = QLabel(f"{self.weak_low_threshold_factor:.2f}")
        weak_low_layout.addWidget(self.weak_low_threshold_slider)
        weak_low_layout.addWidget(self.weak_low_label)

        # High threshold slider (controls both Strong and Weak high thresholds)
        weak_high_layout = QHBoxLayout()
        weak_high_layout.addWidget(QLabel("High Cutoff Percentile:"))
        self.weak_high_threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.weak_high_threshold_slider.setRange(0, 100)
        self.weak_high_threshold_slider.setValue(int(self.weak_high_threshold_factor * 100))
        self.weak_high_threshold_slider.valueChanged.connect(lambda value, name="weak_high": self._update_threshold_sliders(name, value))
        self.weak_high_threshold_slider.sliderReleased.connect(self.update_plots) # Moved here

        self.weak_high_label = QLabel(f"{self.weak_high_threshold_factor:.2f}")
        weak_high_layout.addWidget(self.weak_high_threshold_slider)
        weak_high_layout.addWidget(self.weak_high_label)

        weak_threshold_layout.addLayout(weak_low_layout)
        weak_threshold_layout.addLayout(weak_high_layout)
        weak_threshold_group.setLayout(weak_threshold_layout)

        # General Statistics Label (for detailed region counts, etc.)
        self.common_reflections = QLabel("Common reflections: N/A")
        self.scale_factor = QLabel("Scale factor: N/A")
        sum_display_layout = QHBoxLayout()
        sum_display_layout.addWidget(self.common_reflections)
        sum_display_layout.addWidget(self.scale_factor)
        threshold_layout.addLayout(sum_display_layout)

        # Reflection Count Statistics (Low/Mid/High for Strong dataset - kept for quick overview)
        stats_display_layout = QHBoxLayout()
        self.low_region_label = QLabel("Low: N/A")
        self.middle_region_label = QLabel("Mid: N/A")
        self.high_region_label = QLabel("High: N/A")
        stats_display_layout.addWidget(self.low_region_label)
        stats_display_layout.addWidget(self.middle_region_label)
        stats_display_layout.addWidget(self.high_region_label)
        threshold_layout.addLayout(stats_display_layout)

        threshold_group.setLayout(threshold_layout)
        # Action buttons
        action_layout = QHBoxLayout()
        self.merge_btn = QPushButton("Merge Data")
        self.merge_btn.clicked.connect(self.merge_data)

        self.save_btn = QPushButton("Save Merged Data")
        self.save_btn.clicked.connect(self.save_merged_data)
        self.save_btn.setEnabled(False)

        action_layout.addWidget(self.merge_btn)
        action_layout.addWidget(self.save_btn)

        # Progress bar
        progress_layout = QVBoxLayout()
        progress_label = QLabel("Progress:")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%p%")

        progress_layout.addWidget(progress_label)
        progress_layout.addWidget(self.progress_bar)
          # Add all control layouts to top section
        top_layout.addWidget(file_group)
        top_layout.addWidget(threshold_group)
        top_layout.addWidget(weak_threshold_group)
        # top_layout.addWidget(self.stats_label) # Add stats_label to the top_layout if preferred
        top_layout.addLayout(action_layout)
        top_layout.addLayout(progress_layout)

        # Bottom section: Plots
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)

        # self.canvas = MatplotlibCanvas(width=5, height=4, dpi=100) # Removed redundant canvas creation
        bottom_layout.addWidget(self.canvas)

        # Add widgets to splitter
        splitter.addWidget(top_widget)
        splitter.addWidget(bottom_widget)

        # Set splitter sizes
        splitter.setSizes([300, 500]) # Adjusted for potentially simpler control panel

        # Add splitter to main layout
        main_layout.addWidget(splitter)
        self.setCentralWidget(central_widget)

        # Status bar for messages
        self.statusBar().showMessage("Ready")

    def _get_intensity_at_percentile(self, percentile, sorted_intensities):
        """Calculates the intensity value at a given percentile of sorted intensities."""
        if sorted_intensities is None or len(sorted_intensities) == 0:
            return 0 # Default if no intensities are available

        # Ensure percentile is within [0, 1]
        percentile = np.clip(percentile, 0.0, 1.0)

        index = int(percentile * (len(sorted_intensities) - 1))
        # Clip index to be within valid bounds for the array
        index = np.clip(index, 0, len(sorted_intensities) - 1)
        return sorted_intensities[index]

    def _update_threshold_sliders(self, slider_name, value):
        factor = value / 100.0

        if slider_name == "weak_low":
            # Update both weak and strong low thresholds
            self.weak_low_threshold_factor = factor
            self.low_threshold_factor = factor  # Set the strong threshold to match

            # Ensure low threshold doesn't exceed high threshold
            if self.weak_low_threshold_factor > self.weak_high_threshold_factor:
                self.weak_high_threshold_factor = self.weak_low_threshold_factor
                self.weak_high_threshold_slider.setValue(int(self.weak_high_threshold_factor * 100))
                self.high_threshold_factor = self.weak_high_threshold_factor  # Update strong high threshold too

        elif slider_name == "weak_high":
            # Update both weak and strong high thresholds
            self.weak_high_threshold_factor = factor
            self.high_threshold_factor = factor  # Set the strong threshold to match

            # Ensure high threshold isn't below low threshold
            if self.weak_high_threshold_factor < self.weak_low_threshold_factor:
                self.weak_low_threshold_factor = self.weak_high_threshold_factor
                self.weak_low_threshold_slider.setValue(int(self.weak_low_threshold_factor * 100))
                self.low_threshold_factor = self.weak_low_threshold_factor  # Update strong low threshold too

        # Update labels
        self.weak_low_label.setText(f"{self.weak_low_threshold_factor:.2f}")
        self.weak_high_label.setText(f"{self.weak_high_threshold_factor:.2f}")        # Update both strong and weak thresholds using the same percentile values
        # For strong dataset
        if self.sorted_strong_intensities is not None and self.sorted_strong_intensities.size > 0:
            self.strong_low_threshold_val = self._get_intensity_at_percentile(self.weak_low_threshold_factor, self.sorted_strong_intensities)
            self.strong_high_threshold_val = self._get_intensity_at_percentile(self.weak_high_threshold_factor, self.sorted_strong_intensities)
        else:
            self.strong_low_threshold_val = 0
            self.strong_high_threshold_val = np.inf

        # For weak dataset
        if self.sorted_weak_intensities is not None and self.sorted_weak_intensities.size > 0:
            self.weak_low_threshold_val = self._get_intensity_at_percentile(self.weak_low_threshold_factor, self.sorted_weak_intensities)
            self.weak_high_threshold_val = self._get_intensity_at_percentile(self.weak_high_threshold_factor, self.sorted_weak_intensities)
        else:
            self.weak_low_threshold_val = 0
            self.weak_high_threshold_val = np.inf

        self._update_threshold_info_live()
        self._live_update_plot_elements()
        self.update_status_bar_with_thresholds()

    def _live_update_plot_elements(self):
        """Update only the threshold lines and their labels on both plots."""
        if not hasattr(self, 'canvas') or self.sorted_strong_intensities is None or self.sorted_weak_intensities is None:
            return

        s_low = self.strong_low_threshold_val
        s_high = self.strong_high_threshold_val
        w_low = self.weak_low_threshold_val
        w_high = self.weak_high_threshold_val

        log_sqrt_s_low = np.log10(np.sqrt(s_low)) if s_low > 1e-9 else -np.inf
        log_sqrt_s_high = np.log10(np.sqrt(s_high)) if s_high > 1e-9 else -np.inf
        log_sqrt_w_low = np.log10(np.sqrt(w_low)) if w_low > 1e-9 else -np.inf
        log_sqrt_w_high = np.log10(np.sqrt(w_high)) if w_high > 1e-9 else -np.inf

        # Update Log Plot Lines
        if self.low_thresh_line_log:
            self.low_thresh_line_log.set_ydata([log_sqrt_s_low, log_sqrt_s_low])
            self.low_thresh_line_log.set_label(f'S_Low_L ({s_low:.1f})')
        if self.high_thresh_line_log:
            self.high_thresh_line_log.set_ydata([log_sqrt_s_high, log_sqrt_s_high])
            self.high_thresh_line_log.set_label(f'S_High_L ({s_high:.1f})')
        if self.weak_low_thresh_line_log:
            self.weak_low_thresh_line_log.set_xdata([log_sqrt_w_low, log_sqrt_w_low])
            self.weak_low_thresh_line_log.set_label(f'W_Low_L ({w_low:.1f})')
        if self.weak_high_thresh_line_log:
            self.weak_high_thresh_line_log.set_xdata([log_sqrt_w_high, log_sqrt_w_high])
            self.weak_high_thresh_line_log.set_label(f'W_High_L ({w_high:.1f})')

        # Update Linear Plot Lines
        if self.s_low_line_lin:
            self.s_low_line_lin.set_ydata([s_low, s_low])
            self.s_low_line_lin.set_label(f'S_Low ({s_low:.1f})')
        if self.s_high_line_lin:
            self.s_high_line_lin.set_ydata([s_high, s_high])
            self.s_high_line_lin.set_label(f'S_High ({s_high:.1f})')
        if self.w_low_line_lin:
            self.w_low_line_lin.set_xdata([w_low, w_low])
            self.w_low_line_lin.set_label(f'W_Low ({w_low:.1f})')
        if self.w_high_line_lin:
            self.w_high_line_lin.set_xdata([w_high, w_high])
            self.w_high_line_lin.set_label(f'W_High ({w_high:.1f})')

        # Redraw legends if they exist and have handles
        if self.canvas.ax_log.get_legend() and self.canvas.ax_log.get_legend().legend_handles:
            self.canvas.ax_log.legend(fontsize='small', loc='best')
        if self.canvas.ax_linear.get_legend() and self.canvas.ax_linear.get_legend().legend_handles:
            self.canvas.ax_linear.legend(fontsize='small', loc='best')

        self.canvas.draw_idle()

    def update_status_bar_with_thresholds(self):
        """Update status bar with current threshold values using rich text for proper line breaks"""
        if hasattr(self, 'strong_low_threshold_val') and hasattr(self, 'strong_high_threshold_val'):
            strong_msg = f"Strong thresholds: {self.strong_low_threshold_val:.1f}-{self.strong_high_threshold_val:.1f}"
            weak_msg = ""

            if hasattr(self, 'weak_low_threshold_val') and hasattr(self, 'weak_high_threshold_val'):
                weak_msg = f"Weak thresholds: {self.weak_low_threshold_val:.1f}-{self.weak_high_threshold_val:.1f}"
                  # Add percentile values for clarity
            percentile_info = f"(Percentile range: {self.weak_low_threshold_factor:.2f}-{self.weak_high_threshold_factor:.2f})"
            strong_msg += f" {percentile_info}"

            # First clear any existing widgets
            status_bar = self.statusBar()
            status_bar.clearMessage()

            # Create a temporary widgets list to track what we need to remove
            temp_widgets = []
            for i in range(len(status_bar.children())):
                widget = status_bar.children()[i]
                if isinstance(widget, QLabel) and widget.objectName() == "threshold_status_label":
                    temp_widgets.append(widget)

            # Remove old widgets
            for widget in temp_widgets:
                status_bar.removeWidget(widget)

            # Create a QLabel with properly formatted text
            status_label = QLabel()
            status_label.setObjectName("threshold_status_label")

            if weak_msg:
                status_label.setText(f"{strong_msg}<br>{weak_msg}")
            else:
                status_label.setText(strong_msg)

            # Allow rich text with line breaks
            status_label.setTextFormat(Qt.TextFormat.RichText)
            status_bar.addWidget(status_label)

    def _update_threshold_info_live(self):
        """Updates threshold lines and reflection count labels live during slider drag."""
        # Ensure all plot elements are updated
        self._live_update_plot_elements()

        active_axes = self.canvas.ax_log # Target the log plot for these specific threshold lines

        if self.sorted_strong_intensities is None or len(self.sorted_strong_intensities) == 0:
            # Clear lines if they exist
            if self.low_thresh_line:
                try:
                    self.low_thresh_line.remove()
                except ValueError:
                    pass # Already removed
                self.low_thresh_line = None
            if self.high_thresh_line:
                try:
                    self.high_thresh_line.remove()
                except ValueError:
                    pass # Already removed
                self.high_thresh_line = None

            self.low_region_label.setText("Low: N/A")
            self.middle_region_label.setText("Mid: N/A")
            self.high_region_label.setText("High: N/A")
            if hasattr(active_axes, 'legend') and active_axes.legend_ is not None:
                active_axes.legend()
            self.canvas.draw_idle()
            return

        low_thresh_val = self._get_intensity_at_percentile(self.low_threshold_factor, self.sorted_strong_intensities)
        high_thresh_val = self._get_intensity_at_percentile(self.high_threshold_factor, self.sorted_strong_intensities)

        if low_thresh_val > high_thresh_val: # Ensure correct order if sliders crossed
            low_thresh_val, high_thresh_val = high_thresh_val, low_thresh_val

        # Update threshold lines - these are vertical lines on the X-axis of the log plot (log_sqrt_i_weak)
        # The values low_thresh_val and high_thresh_val are original intensities.
        # For the log plot, these should be transformed if they are to be plotted directly.
        # However, the current implementation in update_plots draws axhline/axvline on log-sqrt values.
        # This function seems to be for the region count labels and their display.
        # The actual lines are drawn in update_plots.

        # Let's keep this function focused on updating labels, and ensure update_plots handles drawing on correct axes.

        # Update count labels based on common reflections (using original strong intensities)
        if self.common_i_strong_np.size > 0:
            low_indices = np.where(self.common_i_strong_np < low_thresh_val)[0]
            middle_indices = np.where((self.common_i_strong_np >= low_thresh_val) & (self.common_i_strong_np <= high_thresh_val))[0]
            high_indices = np.where(self.common_i_strong_np > high_thresh_val)[0]

            self.low_region_label.setText(f"Low (<{low_thresh_val:.1f}): {len(low_indices)}")
            self.middle_region_label.setText(f"Mid [{low_thresh_val:.1f}-{high_thresh_val:.1f}]: {len(middle_indices)}")
            self.high_region_label.setText(f"High (>{high_thresh_val:.1f}): {len(high_indices)}")
            self.common_reflections.setText(f"Common reflections: {len(self.common_i_strong_np)}")
        else:
            self.low_region_label.setText(f"Low (<{low_thresh_val:.1f}): N/A")
            self.middle_region_label.setText(f"Mid [{low_thresh_val:.1f}-{high_thresh_val:.1f}]: N/A")
            self.high_region_label.setText(f"High (>{high_thresh_val:.1f}): N/A")

        # The legend update here might be problematic if lines are on different axes.
        # Let's rely on update_plots to handle legends for each subplot.
        # if hasattr(active_axes, 'legend') and active_axes.legend_ is not None:
        #     handles, labels = active_axes.get_legend_handles_labels()
        #     if handles: # Only update legend if there are items
        #          active_axes.legend(handles, labels, fontsize='small')
        self.canvas.draw_idle() # Redraw to show updated labels if they are part of the canvas (they are not)

    def load_files(self):
        """Load HKL files from disk"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select HKL Files", "", "HKL Files (*.hkl)"
        )

        if not file_paths:
            return

        self.statusBar().showMessage("Loading files...")
        self.reflection_sets = []
        self.file_names = []

        # Clear comboboxes
        self.file_combo1.clear()
        self.file_combo2.clear()

        # Load each file
        for file_path in file_paths:
            try:
                refl_set = read_hkl(file_path)
                self.reflection_sets.append(refl_set)
                self.file_names.append(os.path.basename(file_path))

                # Add to comboboxes
                self.file_combo1.addItem(os.path.basename(file_path))
                self.file_combo2.addItem(os.path.basename(file_path))

            except Exception as e:
                self.statusBar().showMessage(f"Error loading {os.path.basename(file_path)}: {str(e)}")

        if len(self.reflection_sets) >= 2:
            # Default selection: first two files
            self.file_combo1.setCurrentIndex(0)
            self.file_combo2.setCurrentIndex(1)
        elif len(self.reflection_sets) == 1:
            self.file_combo1.setCurrentIndex(0)
            # Potentially disable file_combo2 or prompt for a second file
            self.file_combo2.setEnabled(False) # Disable until another file is loaded

        self.statusBar().showMessage(f"Loaded {len(self.reflection_sets)} HKL files")
        self.update_plots() # This will do a full redraw and set up threshold lines

    def set_actual_strong_and_weak(self, strong_idx, weak_idx):
        if strong_idx != weak_idx and 0 <= strong_idx < self.file_combo1.count() and 0 <= weak_idx < self.file_combo2.count():
            self.file_combo1.setCurrentIndex(strong_idx)
            self.file_combo2.setCurrentIndex(weak_idx)

    def update_plots(self):
        """Update the visualization plot: log10(sqrt(I_strong)) vs log10(sqrt(I_weak)) for common HKLs."""
        self.canvas.ax_log.clear()
        self.canvas.ax_linear.clear()

        # Reset threshold lines for both plots
        self.low_thresh_line_log = None
        self.high_thresh_line_log = None
        self.weak_low_thresh_line_log = None
        self.weak_high_thresh_line_log = None

        self.s_low_line_lin = None # For linear plot: Strong Low Threshold
        self.s_high_line_lin = None # For linear plot: Strong High Threshold
        self.w_low_line_lin = None  # For linear plot: Weak Low Threshold
        self.w_high_line_lin = None # For linear plot: Weak High Threshold

        self.sorted_strong_intensities = None
        self.sorted_weak_intensities = None

        if len(self.reflection_sets) < 2:
            self.canvas.ax_log.text(0.5, 0.5, "Please load at least two HKL files.",
                                  horizontalalignment='center', verticalalignment='center')
            self.canvas.ax_linear.text(0.5, 0.5, "Load files to see linear plot.",
                                  horizontalalignment='center', verticalalignment='center')
            self.low_region_label.setText("Low: N/A")
            self.middle_region_label.setText("Mid: N/A")
            self.high_region_label.setText("High: N/A")
            self.canvas.ax_log.set_xlabel("log10(sqrt(I_weak))")
            self.canvas.ax_log.set_ylabel("log10(sqrt(I_strong))")
            self.canvas.ax_linear.set_xlabel("I_weak")
            self.canvas.ax_linear.set_ylabel("I_strong")
            self.canvas.draw()
            return

        idx1 = self.file_combo1.currentIndex()
        idx2 = self.file_combo2.currentIndex() # Corrected 'this' to 'self'

        if idx1 < 0 or idx2 < 0 or idx1 == idx2 or idx1 >= len(self.reflection_sets) or idx2 >= len(self.reflection_sets):
            self.canvas.ax_log.text(0.5, 0.5, "Please select two different HKL files.",
                                  horizontalalignment='center', verticalalignment='center')
            self.canvas.ax_linear.text(0.5, 0.5, "Select files to see linear plot.",
                                  horizontalalignment='center', verticalalignment='center')
            self.low_region_label.setText("Low: N/A")
            self.middle_region_label.setText("Mid: N/A")
            self.high_region_label.setText("High: N/A")
            self.canvas.ax_log.set_xlabel("log10(sqrt(I_weak))")
            self.canvas.ax_log.set_ylabel("log10(sqrt(I_strong))")
            self.canvas.ax_linear.set_xlabel("I_weak")
            self.canvas.ax_linear.set_ylabel("I_strong")
            self.canvas.draw()
            return

        set1 = self.reflection_sets[idx1]
        set2 = self.reflection_sets[idx2]
        name1 = self.file_names[idx1]
        name2 = self.file_names[idx2]

        # Auto-detect strong and weak datasets
        if not set1._data_finalized:
            set1._finalize_data_structure()
        if not set2._data_finalized:
            set2._finalize_data_structure()

        max_i1 = np.max(set1.refl_list_np[1]) if set1.refl_list_np[1].size > 0 else -np.inf
        max_i2 = np.max(set2.refl_list_np[1]) if set2.refl_list_np[1].size > 0 else -np.inf

        if max_i1 >= max_i2:
            actual_strong_set, actual_weak_set = set1, set2
            actual_strong_name, actual_weak_name = name1, name2
            strong_idx, weak_idx = idx1, idx2
        else:
            actual_strong_set, actual_weak_set = set2, set1
            actual_strong_name, actual_weak_name = name2, name1
            strong_idx, weak_idx = idx2, idx1
        # Ensure combos reflect the actual strong/weak
        self.set_actual_strong_and_weak(strong_idx, weak_idx)

        if actual_strong_set.refl_list_np[1].size > 0:
            self.sorted_strong_intensities = np.sort(actual_strong_set.refl_list_np[1])
        else:
            self.statusBar().showMessage(f"Warning: Strong dataset ({actual_strong_name}) has no intensity data.")
            self.sorted_strong_intensities = np.array([])

        if actual_weak_set.refl_list_np[1].size > 0:
            self.sorted_weak_intensities = np.sort(actual_weak_set.refl_list_np[1])
        else:
            # self.statusBar().showMessage(f"Warning: Weak dataset ({actual_weak_name}) has no intensity data.") # Can be noisy
            self.sorted_weak_intensities = np.array([])

        # Calculate absolute threshold values based on current percentile factors
        self.strong_low_threshold_val = self._get_intensity_at_percentile(self.low_threshold_factor, self.sorted_strong_intensities)
        self.strong_high_threshold_val = self._get_intensity_at_percentile(self.high_threshold_factor, self.sorted_strong_intensities)
        # For plot symmetry, calculate weak thresholds using the same percentile factors on the weak dataset
        self.weak_low_threshold_val = self._get_intensity_at_percentile(self.low_threshold_factor, self.sorted_weak_intensities)
        self.weak_high_threshold_val = self._get_intensity_at_percentile(self.high_threshold_factor, self.sorted_weak_intensities)

        # Get common reflections (original intensities)
        common_hkls, i_strong_common_orig, _, i_weak_common_orig, _ = actual_strong_set.get_common_reflections(actual_weak_set)

        if common_hkls.size == 0:
            self.canvas.ax_log.text(0.5, 0.5, "No common HKLs found.", horizontalalignment='center', verticalalignment='center')
            self.canvas.ax_linear.text(0.5, 0.5, "No common HKLs for linear plot.", horizontalalignment='center', verticalalignment='center')
            self.low_region_label.setText("Low: N/A")
            self.middle_region_label.setText("Mid: N/A")
            self.high_region_label.setText("High: N/A")
            self.canvas.ax_log.set_xlabel("log10(sqrt(I_weak))")
            self.canvas.ax_log.set_ylabel("log10(sqrt(I_strong))")
            self.canvas.ax_linear.set_xlabel("I_weak")
            self.canvas.ax_linear.set_ylabel("I_strong")
            self.canvas.draw()
            return

        # --- Log Plot Calculations ---
        # Mask out negative values before sqrt to avoid RuntimeWarning
        valid_mask = (i_strong_common_orig >= 0) & (i_weak_common_orig >= 0)
        i_strong_common_orig = i_strong_common_orig[valid_mask]
        i_weak_common_orig = i_weak_common_orig[valid_mask]
        sqrt_i_strong_common = np.sqrt(i_strong_common_orig)
        sqrt_i_weak_common = np.sqrt(i_weak_common_orig)

        # Filter for positive sqrt intensities before log transformation
        positive_mask = (sqrt_i_strong_common > 1e-9) & (sqrt_i_weak_common > 1e-9) # Use a small epsilon

        if not np.any(positive_mask):
            self.canvas.ax_log.text(0.5, 0.5, "No common HKLs with positive intensities for log plot.", horizontalalignment='center', verticalalignment='center')
            # Linear plot can still be made with original intensities
        else:
            # Proceed with log plot if data exists
            log_sqrt_i_strong = np.log10(sqrt_i_strong_common[positive_mask])
            log_sqrt_i_weak = np.log10(sqrt_i_weak_common[positive_mask])

            self.common_i_strong_np = i_strong_common_orig # Used by _update_threshold_info_live

            log_sqrt_s_low = np.log10(np.sqrt(self.strong_low_threshold_val)) if self.strong_low_threshold_val > 1e-9 else -np.inf
            log_sqrt_s_high = np.log10(np.sqrt(self.strong_high_threshold_val)) if self.strong_high_threshold_val > 1e-9 else -np.inf
            log_sqrt_w_low = np.log10(np.sqrt(self.weak_low_threshold_val)) if self.weak_low_threshold_val > 1e-9 else -np.inf
            log_sqrt_w_high = np.log10(np.sqrt(self.weak_high_threshold_val)) if self.weak_high_threshold_val > 1e-9 else -np.inf

            # Define new region masks for three regions (mutually exclusive, all points assigned)
            region_both_log = (
                (log_sqrt_i_strong >= log_sqrt_s_low) & (log_sqrt_i_strong < log_sqrt_s_high) &
                (log_sqrt_i_weak >= log_sqrt_w_low) & (log_sqrt_i_weak < log_sqrt_w_high)
            )
            region_strong_log = (~region_both_log) & ((log_sqrt_i_strong < log_sqrt_s_low) | (log_sqrt_i_weak < log_sqrt_w_low))
            region_weak_log = ~(region_both_log | region_strong_log)  # All other points

            # Plot the three regions
            self.canvas.ax_log.scatter(log_sqrt_i_weak[region_strong_log], log_sqrt_i_strong[region_strong_log], c=self.COLOR_USE_STRONG, alpha=0.5, label='Use Strong data', s=10)
            self.canvas.ax_log.scatter(log_sqrt_i_weak[region_both_log], log_sqrt_i_strong[region_both_log], c=self.COLOR_USE_BOTH, alpha=0.7, label='Use both data', s=10)
            self.canvas.ax_log.scatter(log_sqrt_i_weak[region_weak_log], log_sqrt_i_strong[region_weak_log], c=self.COLOR_USE_WEAK, alpha=0.7, label='Use weak data', s=10)

            if np.isfinite(log_sqrt_s_low):
                self.low_thresh_line_log = self.canvas.ax_log.axhline(log_sqrt_s_low, color='cyan', linestyle='--', label=f'S_Low_L ({self.strong_low_threshold_val:.1f})')
            if np.isfinite(log_sqrt_s_high):
                self.high_thresh_line_log = self.canvas.ax_log.axhline(log_sqrt_s_high, color='magenta', linestyle='--', label=f'S_High_L ({self.strong_high_threshold_val:.1f})')
            if np.isfinite(log_sqrt_w_low):
                self.weak_low_thresh_line_log = self.canvas.ax_log.axvline(log_sqrt_w_low, color='lime', linestyle='--', label=f'W_Low_L ({self.weak_low_threshold_val:.1f})')
            if np.isfinite(log_sqrt_w_high):
                self.weak_high_thresh_line_log = self.canvas.ax_log.axvline(log_sqrt_w_high, color='orange', linestyle='--', label=f'W_High_L ({self.weak_high_threshold_val:.1f})')

            self.canvas.ax_log.set_xlabel(f"log10(sqrt(I_weak)) - {actual_weak_name}")
            self.canvas.ax_log.set_ylabel(f"log10(sqrt(I_strong)) - {actual_strong_name}")
            self.canvas.ax_log.set_title("Log-Log Sqrt(Intensity) Plot")
            self.canvas.ax_log.grid(True, linestyle=':', alpha=0.3)
            # Remove previous legend if it exists
            if self.canvas.ax_log.get_legend() is not None:
                self.canvas.ax_log.get_legend().remove()
            handles_log, labels_log = self.canvas.ax_log.get_legend_handles_labels()
            if handles_log:
                self.canvas.ax_log.legend(handles_log, labels_log, fontsize='small', loc='best')

        # --- Linear Plot Calculations & Drawing ---
        # Use original intensities: i_strong_common_orig, i_weak_common_orig
        # Define regions for linear plot based on original intensity thresholds
        s_low_lin_thresh = self.strong_low_threshold_val
        s_high_lin_thresh = self.strong_high_threshold_val
        w_low_lin_thresh = self.weak_low_threshold_val
        w_high_lin_thresh = self.weak_high_threshold_val

        # Filter out non-positive intensities for linear plot if needed
        # For regression through origin, ensure positive values if that's a constraint of the method.
        valid_linear_mask = (i_strong_common_orig > 1e-9) & (i_weak_common_orig > 1e-9)

        i_strong_lin = i_strong_common_orig[valid_linear_mask]
        i_weak_lin = i_weak_common_orig[valid_linear_mask]

        if i_strong_lin.size > 0 and i_weak_lin.size > 0:
            # Define new region masks for three regions (mutually exclusive, all points assigned)
            region_both_lin = (
                (i_strong_lin >= s_low_lin_thresh) & (i_strong_lin < s_high_lin_thresh) &
                (i_weak_lin >= w_low_lin_thresh) & (i_weak_lin < w_high_lin_thresh)
            )
            region_strong_lin = (~region_both_lin) & ((i_strong_lin < s_low_lin_thresh) | (i_weak_lin < w_low_lin_thresh))
            region_weak_lin = ~(region_both_lin | region_strong_lin)  # All other points

            self.canvas.ax_linear.scatter(i_weak_lin[region_strong_lin], i_strong_lin[region_strong_lin], c=self.COLOR_USE_STRONG, alpha=0.3, label='Use Strong data', s=10)
            self.canvas.ax_linear.scatter(i_weak_lin[region_both_lin], i_strong_lin[region_both_lin], c=self.COLOR_USE_BOTH, alpha=0.5, label='Use both data', s=10)
            self.canvas.ax_linear.scatter(i_weak_lin[region_weak_lin], i_strong_lin[region_weak_lin], c=self.COLOR_USE_WEAK, alpha=0.5, label='Use weak data', s=10)


            # Linear regression through the origin: Is = slope_lin * Iw
            # Use points from 'Region 1 (Linear)' for this regression
            i_s_reg_lin = i_strong_lin[region_both_lin]
            i_w_reg_lin = i_weak_lin[region_both_lin]

            if i_s_reg_lin.size > 0 and i_w_reg_lin.size > 0:
                # Forcing through origin: slope = sum(x*y) / sum(x^2)
                # Here, x = i_w_reg_lin, y = i_s_reg_lin
                slope_lin_origin = np.sum(i_w_reg_lin * i_s_reg_lin) / np.sum(i_w_reg_lin**2)
                self.linear_regression_slope_origin = slope_lin_origin # Store for merging

                max_iw_lin = np.max(i_weak_lin) if i_weak_lin.size > 0 else 0
                reg_x_lin_vals = np.array([0, max_iw_lin])
                self.canvas.ax_linear.plot(reg_x_lin_vals, slope_lin_origin * reg_x_lin_vals, 'green', linestyle='--', label=f'LinReg (Origin, m={slope_lin_origin:.2f})')
            else:
                self.linear_regression_slope_origin = 1.0 # Default
            self.scale_factor.setText(f"Scale factor (slope): {self.linear_regression_slope_origin:.2f}")

            # Threshold lines for linear plot - ENABLED
            if np.isfinite(s_low_lin_thresh):
                self.s_low_line_lin = self.canvas.ax_linear.axhline(s_low_lin_thresh, color='cyan', linestyle=':', alpha=0.7, label=f'S_Low ({s_low_lin_thresh:.1f})')
            if np.isfinite(s_high_lin_thresh):
                self.s_high_line_lin = self.canvas.ax_linear.axhline(s_high_lin_thresh, color='magenta', linestyle=':', alpha=0.7, label=f'S_High ({s_high_lin_thresh:.1f})')
            if np.isfinite(w_low_lin_thresh):
                self.w_low_line_lin = self.canvas.ax_linear.axvline(w_low_lin_thresh, color='lime', linestyle=':', alpha=0.7, label=f'W_Low ({w_low_lin_thresh:.1f})')
            if np.isfinite(w_high_lin_thresh):
                self.w_high_line_lin = self.canvas.ax_linear.axvline(w_high_lin_thresh, color='orange', linestyle=':', alpha=0.7, label=f'W_High ({w_high_lin_thresh:.1f})')

            self.canvas.ax_linear.set_xlabel(f"I_weak - {actual_weak_name}")
            self.canvas.ax_linear.set_ylabel(f"I_strong - {actual_strong_name}")
            self.canvas.ax_linear.set_title("Linear Intensity Plot")
            self.canvas.ax_linear.grid(True, linestyle=':', alpha=0.3)
            # Remove previous legend if it exists
            if self.canvas.ax_linear.get_legend() is not None:
                self.canvas.ax_linear.get_legend().remove()
            handles_lin, labels_lin = self.canvas.ax_linear.get_legend_handles_labels()
            if handles_lin:
                self.canvas.ax_linear.legend(handles_lin, labels_lin, fontsize='small', loc='best')
        else:
            self.canvas.ax_linear.text(0.5, 0.5, "No valid data for linear plot.", horizontalalignment='center', verticalalignment='center')
            self.canvas.ax_linear.set_xlabel(f"I_weak - {actual_weak_name}")
            self.canvas.ax_linear.set_ylabel(f"I_strong - {actual_strong_name}")
            self.canvas.ax_linear.set_title("Linear Intensity Plot")

        self._update_threshold_info_live()

        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def merge_data(self):
        """Merge reflection data: include all original reflections from both files, preserving multiplicity. For common HKLs, write both the strong and scaled weak entries to the merged file (doubled multiplicity in overlap)."""
        if len(self.reflection_sets) < 2:
            self.statusBar().showMessage("Need at least two data sets to merge")
            return

        idx1 = self.file_combo1.currentIndex()
        idx2 = self.file_combo2.currentIndex()

        if idx1 < 0 or idx2 < 0 or idx1 == idx2 or idx1 >= len(self.reflection_sets) or idx2 >= len(self.reflection_sets):
            self.statusBar().showMessage("Please select two different HKL files for merging.")
            return

        set1_orig = self.reflection_sets[idx1]
        set2_orig = self.reflection_sets[idx2]

        # Auto-detect strong and weak datasets for merging consistency
        max_i1 = np.max(set1_orig.refl_list_np[1]) if set1_orig.refl_list_np[1].size > 0 else -np.inf
        max_i2 = np.max(set2_orig.refl_list_np[1]) if set2_orig.refl_list_np[1].size > 0 else -np.inf

        if max_i1 >= max_i2:
            actual_strong_set, actual_weak_set = set1_orig, set2_orig
            actual_strong_name, _ = self.file_names[idx1], self.file_names[idx2]
        else:
            actual_strong_set, actual_weak_set = set2_orig, set1_orig
            actual_strong_name, _ = self.file_names[idx2], self.file_names[idx1]

        self.progress_bar.setValue(0)
        self.statusBar().showMessage(f"Setting up merge: Strong={actual_strong_name} (auto)")
        QCoreApplication.processEvents()

        self.merged_data = reflection_list()
        if not actual_strong_set._data_finalized:
            actual_strong_set._finalize_data_structure()
        if not actual_weak_set._data_finalized:
            actual_weak_set._finalize_data_structure()

        if actual_strong_set.cell:
            self.merged_data.set_cell(actual_strong_set.cell.a, actual_strong_set.cell.b, actual_strong_set.cell.c,
                                      actual_strong_set.cell.alpha, actual_strong_set.cell.beta, actual_strong_set.cell.gamma,
                                      wavelength=actual_strong_set.cell.wavelength)
        elif actual_weak_set.cell:
            self.merged_data.set_cell(actual_weak_set.cell.a, actual_weak_set.cell.b, actual_weak_set.cell.c,
                                      actual_weak_set.cell.alpha, actual_weak_set.cell.beta, actual_weak_set.cell.gamma,
                                      wavelength=actual_weak_set.cell.wavelength)

        merge_scale_factor = self.linear_regression_slope_origin if hasattr(self, 'linear_regression_slope_origin') and self.linear_regression_slope_origin is not None else 1.0

        # 1. Add all strong entries (preserve multiplicity)
        for i in range(actual_strong_set.refl_list_np[0].shape[0]):
            intensity = actual_strong_set.refl_list_np[1][i]
            if intensity > self.weak_high_threshold_val:
                continue
            h, k, l = actual_strong_set.refl_list_np[0][i]
            
            sigma = actual_strong_set.refl_list_np[2][i]
            self.merged_data.append(h, k, l, intensity, sigma)

        # 2. Add all weak entries above the Low cutoff (preserve multiplicity, scale intensity/sigma)
        for i in range(actual_weak_set.refl_list_np[0].shape[0]):
            intensity = actual_weak_set.refl_list_np[1][i] * merge_scale_factor
            if intensity < self.weak_low_threshold_val:
                continue
            h, k, l = actual_weak_set.refl_list_np[0][i]
            sigma = actual_weak_set.refl_list_np[2][i] * merge_scale_factor
            self.merged_data.append(h, k, l, intensity, sigma)

        self.merged_data._finalize_data_structure()
        self.progress_bar.setValue(self.merged_data.refl_list_np[0].shape[0])
        self.statusBar().showMessage(f"Merge complete. Merged: {self.merged_data.refl_list_np[0].shape[0]} reflections (multiplicity preserved, overlap doubled)")
        self.save_btn.setEnabled(True)

    def save_merged_data(self):
        """Save the merged reflection data to an HKL file"""
        if self.merged_data is None:
            self.statusBar().showMessage("No merged data to save.")
            return

        # Ensure data is finalized before accessing refl_list_np
        if not hasattr(self.merged_data, 'refl_list_np') or not self.merged_data._data_finalized:
            self.merged_data._finalize_data_structure()

        # Now check if there's any data
        if self.merged_data.refl_list_np[0].size == 0:
            self.statusBar().showMessage("No merged data to save.")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Merged HKL File", "merged_data.hkl", "HKL Files (*.hkl)"
        )

        if not file_path:
            return

        try:
            with open(file_path, 'w') as f:
                # Finalize data before accessing refl_list_np
                if not self.merged_data._data_finalized:
                    self.merged_data._finalize_data_structure()

                hkls = self.merged_data.refl_list_np[0]
                intensities = self.merged_data.refl_list_np[1]
                sigmas = self.merged_data.refl_list_np[2]

                # Debug: Print first 10 HKL, intensity, sigma values before writing
                # (debug prints removed)

                for i in range(hkls.shape[0]):
                    hkl = hkls[i]
                    intensity = intensities[i]
                    sigma = sigmas[i]
                    # Format: HHH KKK LLL FFFFFF.FF SSSSSS.SS (adjust spacing as per standard HKL)
                    # Typical SHELX HKL format: (3I4,2F8.2,I4) for H K L I SIGI BATCH
                    # We only have H K L I SIGI. Assuming batch is 0 or 1.
                    if abs(intensity) < 10.0:
                        intensity_string = f"{intensity:8.5f}"
                    elif abs(intensity) < 100.0:
                        intensity_string = f"{intensity:8.4f}"
                    elif abs(intensity) < 1000.0:
                        intensity_string = f"{intensity:8.3f}"
                    elif abs(intensity) < 10000.0:
                        intensity_string = f"{intensity:8.2f}"
                    elif abs(intensity) < 100000.0:
                        intensity_string = f"{intensity:8.1f}"
                    else:
                        intensity_string = f"{intensity:8.0f}"
                    if sigma < 10.0:
                        sigma_string = f"{sigma:8.5f}"
                    elif sigma < 100.0:
                        sigma_string = f"{sigma:8.4f}"
                    elif sigma < 1000.0:
                        sigma_string = f"{sigma:8.3f}"
                    elif sigma < 10000.0:
                        sigma_string = f"{sigma:8.2f}"
                    elif sigma < 100000.0:
                        sigma_string = f"{sigma:8.1f}"
                    else:
                        sigma_string = f"{sigma:8.0f}"
                    f.write(f"{hkl[0]:4d}{hkl[1]:4d}{hkl[2]:4d}{intensity_string}{sigma_string}   1\n") # Added batch 1

                # End of file marker for SHELX
                f.write("   0   0   0    0.00    0.00   0\n")

            self.statusBar().showMessage(f"Merged data saved to {file_path}")
        except Exception as e:
            self.statusBar().showMessage(f"Error saving file: {str(e)}")

    def swap_files(self):
        """Swap the selected files in the comboboxes"""
        idx1 = self.file_combo1.currentIndex()
        idx2 = self.file_combo2.currentIndex()
        if idx1 != idx2 and idx1 >= 0 and idx2 >= 0:
            self.file_combo1.setCurrentIndex(idx2)
            self.file_combo2.setCurrentIndex(idx1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = AttenuatoriumGUI()
    main_win.show()
    sys.exit(app.exec())
