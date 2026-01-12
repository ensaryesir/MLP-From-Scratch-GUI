import customtkinter as ctk
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from config import COLOR_PALETTE as COLOR_DIGITS, COLOR_DATA_POINTS, COLOR_REGRESSION_LINE


class VisualizationFrame(ctk.CTkFrame):
    def __init__(self, master, on_point_added_callback=None, **kwargs):
        super().__init__(master, **kwargs)

        self.on_point_added_callback = on_point_added_callback
        self.loss_history = []
        self.ae_loss_history = []
        self.mlp_loss_history = []
        self.current_task = 'classification'
        self.click_enabled = True
        self.click_handler_id = None

        self.tabview = ctk.CTkTabview(self, width=700, height=600)
        self.tabview.pack(fill="both", expand=True, padx=10, pady=10)

        self.current_dataset_mode = 'manual'
        self.current_model_type = 'MLP'
        self._add_tabs_for_mode('manual')

        self._setup_train_tab()
        self._setup_test_tab()
        self._setup_loss_tab()

    def _setup_train_tab(self):
        tab = self.tabview.tab("🎯 Training")

        self.train_fig = Figure(figsize=(7, 6), dpi=100)
        self.train_ax = self.train_fig.add_subplot(111)
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Training Data - Click to Add Points')
        self.train_ax.grid(True, alpha=0.3)

        self.train_canvas = FigureCanvasTkAgg(self.train_fig, tab)
        self.train_canvas.draw()
        self.train_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.click_handler_id = self.train_canvas.mpl_connect('button_press_event', self._on_train_click)

    def _setup_test_tab(self):
        tab = self.tabview.tab("📊 Test")

        self.test_fig = Figure(figsize=(7, 6), dpi=100)
        self.test_ax = self.test_fig.add_subplot(111)
        self.test_ax.set_xlim(-1, 11)
        self.test_ax.set_ylim(-1, 11)
        self.test_ax.set_xlabel('X')
        self.test_ax.set_ylabel('Y')
        self.test_ax.set_title('Test Data and Model Performance')
        self.test_ax.grid(True, alpha=0.3)

        self.test_canvas = FigureCanvasTkAgg(self.test_fig, tab)
        self.test_canvas.draw()
        self.test_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _setup_loss_tab(self):
        tab = self.tabview.tab("📈 Error Graph")

        self.loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.loss_ax = self.loss_fig.add_subplot(111)
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Error')
        self.loss_ax.set_title('Training Error')
        self.loss_ax.grid(True, alpha=0.3)

        self.loss_canvas = FigureCanvasTkAgg(self.loss_fig, tab)
        self.loss_canvas.draw()
        self.loss_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_ae_loss_tab(self):
        tab = self.tabview.tab("📈 Autoencoder Error")

        self.ae_loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.ae_loss_ax = self.ae_loss_fig.add_subplot(111)
        self.ae_loss_ax.set_xlabel('Epoch')
        self.ae_loss_ax.set_ylabel('Error')
        self.ae_loss_ax.set_title('Autoencoder Training Error')
        self.ae_loss_ax.grid(True, alpha=0.3)

        self.ae_loss_canvas = FigureCanvasTkAgg(self.ae_loss_fig, tab)
        self.ae_loss_canvas.draw()
        self.ae_loss_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _setup_mlp_loss_tab(self):
        tab = self.tabview.tab("📈 MLP Error")

        self.mlp_loss_fig = Figure(figsize=(7, 6), dpi=100)
        self.mlp_loss_ax = self.mlp_loss_fig.add_subplot(111)
        self.mlp_loss_ax.set_xlabel('Epoch')
        self.mlp_loss_ax.set_ylabel('Error')
        self.mlp_loss_ax.set_title('MLP Training Error')
        self.mlp_loss_ax.grid(True, alpha=0.3)

        self.mlp_loss_canvas = FigureCanvasTkAgg(self.mlp_loss_fig, tab)
        self.mlp_loss_canvas.draw()
        self.mlp_loss_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _on_train_click(self, event):
        if not self.click_enabled:
            return
        
        if event.inaxes == self.train_ax and event.xdata and event.ydata:
            x, y = event.xdata, event.ydata
            if self.on_point_added_callback:
                self.on_point_added_callback(x, y)

    def plot_data_points(self, data_handler, ax=None, task='classification'):
        if ax is None:
            ax = self.train_ax

        if task == 'regression':
            all_points = data_handler.get_all_points()
            if len(all_points) > 0:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                ax.scatter(x_coords, y_coords,
                          c=COLOR_DATA_POINTS, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1.5,
                          label='Data Points')
        else:
            for class_id in range(data_handler.get_num_classes()):
                points = data_handler.get_data_by_class(class_id)
                if len(points) > 0:
                    color = data_handler.get_color(class_id)
                    x_coords = [p[0] for p in points]
                    y_coords = [p[1] for p in points]
                    ax.scatter(x_coords, y_coords,
                              c=color, s=100, alpha=0.7,
                              edgecolors='black', linewidth=1.5,
                              label=data_handler.classes[class_id])

        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right')

    def update_train_view(self, data_handler):
        self.train_ax.clear()
        self.train_ax.set_xlim(-1, 11)
        self.train_ax.set_ylim(-1, 11)
        self.train_ax.set_xlabel('X')
        self.train_ax.set_ylabel('Y')
        self.train_ax.set_title('Training Data - Click to Add Points')
        self.train_ax.grid(True, alpha=0.3)

        task = getattr(self, 'current_task', 'classification')
        self.plot_data_points(data_handler, self.train_ax, task=task)

        self.train_canvas.draw()

    def clear_test_view(self):
        self.test_ax.clear()
        self.test_ax.set_xlim(-1, 11)
        self.test_ax.set_ylim(-1, 11)
        self.test_ax.set_xlabel('X')
        self.test_ax.set_ylabel('Y')
        self.test_ax.set_title('Test Data and Model Performance')
        self.test_ax.grid(True, alpha=0.3)
        self.test_canvas.draw()

    def _create_meshgrid(self, x_min=-1, x_max=11, y_min=-1, y_max=11, h=0.1):
        x_range = []
        x_val = x_min
        while x_val < x_max:
            x_range.append(x_val)
            x_val += h
        
        y_range = []
        y_val = y_min
        while y_val < y_max:
            y_range.append(y_val)
            y_val += h
        
        xx = []
        yy = []
        for y_v in y_range:
            xx_row = []
            yy_row = []
            for x_v in x_range:
                xx_row.append(x_v)
                yy_row.append(y_v)
            xx.append(xx_row)
            yy.append(yy_row)
        return xx, yy

    def update_decision_boundary(self, model, X, y, data_handler, tab_name='train', task='classification'):
        if tab_name == 'train':
            ax = self.train_ax
            canvas = self.train_canvas
            title = 'Training Data - Decision Boundaries' if task == 'classification' else 'Training Data - Regression'
        else:
            ax = self.test_ax
            canvas = self.test_canvas
            title = 'Test Data - Model Performance' if task == 'classification' else 'Test Data - Regression'
        
        ax.clear()
        
        if task == 'regression':
            if len(X) == 0:
                canvas.draw()
                return
            
            input_dim = len(X[0]) if len(X) > 0 else 0
            
            if input_dim == 1:
                x_vals = [x[0] for x in X]
                y_vals = y
                
                sorted_pairs = sorted(zip(x_vals, y_vals))
                x_sorted = [p[0] for p in sorted_pairs]
                y_sorted = [p[1] for p in sorted_pairs]
                
                ax.scatter(x_vals, y_vals, c=COLOR_DATA_POINTS, s=100, alpha=0.7,
                          edgecolors='black', linewidth=1.5, label='Data Points', zorder=3)
                
                x_min, x_max = -1, 11
                x_pred_range = []
                x_val = x_min
                h = 0.1
                while x_val <= x_max:
                    x_pred_range.append(x_val)
                    x_val += h
                
                grid_points = [[x] for x in x_pred_range]
                y_pred = model.predict(grid_points)
                
                if isinstance(y_pred[0], (list, tuple)):
                    y_pred = [p[0] if len(p) > 0 else 0.0 for p in y_pred]
                
                ax.plot(x_pred_range, y_pred, COLOR_REGRESSION_LINE, linewidth=2, label='Model Prediction', zorder=2)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(-1, 11)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                
            else:
                x_min, x_max = -1, 11
                y_min, y_max = -1, 11
                h = 0.1
                
                xx, yy = self._create_meshgrid(x_min, x_max, y_min, y_max, h)
                
                grid_points = []
                for i in range(len(xx)):
                    for j in range(len(xx[0])):
                        grid_points.append([xx[i][j], yy[i][j]])
                
                Z = model.predict(grid_points)
                
                if isinstance(Z[0], (list, tuple)):
                    Z = [z[0] if len(z) > 0 else 0.0 for z in Z]
                
                Z_grid = []
                idx = 0
                for i in range(len(xx)):
                    Z_row = []
                    for j in range(len(xx[0])):
                        Z_row.append(Z[idx])
                        idx += 1
                    Z_grid.append(Z_row)
                
                colors_list = ['#8B00FF', '#4B0082', '#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FFD700']
                n_bins = 100
                cmap = mcolors.LinearSegmentedColormap.from_list('purple_yellow', colors_list, N=n_bins)
                
                im = ax.contourf(xx, yy, Z_grid, levels=50, cmap=cmap, alpha=0.6)
                ax.figure.colorbar(im, ax=ax, label='Predicted Value')
                
                x_coords = [x[0] for x in X]
                y_coords = [x[1] for x in X]
                ax.scatter(x_coords, y_coords, c='black', s=100, alpha=0.8,
                          edgecolors='white', linewidth=1.5, label='Data Points', zorder=3)
                
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
            
        else:
            x_min, x_max = -1, 11
            y_min, y_max = -1, 11
            h = 0.1
            
            xx, yy = self._create_meshgrid(x_min, x_max, y_min, y_max, h)
            
            grid_points = []
            for i in range(len(xx)):
                for j in range(len(xx[0])):
                    grid_points.append([xx[i][j], yy[i][j]])
            
            Z = model.predict(grid_points)
            
            Z_grid = []
            idx = 0
            for i in range(len(xx)):
                Z_row = []
                for j in range(len(xx[0])):
                    Z_row.append(Z[idx])
                    idx += 1
                Z_grid.append(Z_row)
            
            n_classes = data_handler.get_num_classes()
            colors = [data_handler.get_color(i) for i in range(n_classes)]
            
            levels = [i - 0.5 for i in range(n_classes + 1)]
            ax.contourf(xx, yy, Z_grid, alpha=0.3, levels=levels, colors=colors)
            
            if len(X) > 0:
                for class_id in range(n_classes):
                    mask = [y[i] == class_id for i in range(len(y))]
                    if any(mask):
                        color = data_handler.get_color(class_id)
                        x_coords = [X[i][0] for i in range(len(X)) if mask[i]]
                        y_coords = [X[i][1] for i in range(len(X)) if mask[i]]
                        ax.scatter(x_coords, y_coords,
                                 c=color, s=100, alpha=0.8,
                                 edgecolors='black', linewidth=1.5,
                                 label=data_handler.classes[class_id])
            
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(loc='upper right')
        
        canvas.draw()

    def update_loss_plot(self, epoch, train_loss, model_type='MLP'):
        if model_type == 'Autoencoder':
            self.ae_loss_history.append((epoch, train_loss))
            self.ae_loss_ax.clear()

            if len(self.ae_loss_history) > 0:
                epochs = [e for e, _ in self.ae_loss_history]
                train_losses = [tl for _, tl in self.ae_loss_history]

                self.ae_loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='Autoencoder Error',
                )
                handles, _ = self.ae_loss_ax.get_legend_handles_labels()
                if handles:
                    self.ae_loss_ax.legend(loc='upper right')

            self.ae_loss_ax.grid(True, alpha=0.3)
            self.ae_loss_ax.set_title("Autoencoder Training Loss")
            self.ae_loss_ax.set_xlabel("Epoch")
            self.ae_loss_ax.set_ylabel("Loss")

            self.ae_loss_canvas.draw()
            self.ae_loss_canvas.flush_events()
        
        elif model_type in ['AutoencoderMLP']:
            self.mlp_loss_history.append((epoch, train_loss))
            self.mlp_loss_ax.clear()

            if len(self.mlp_loss_history) > 0:
                epochs = [e for e, _ in self.mlp_loss_history]
                train_losses = [tl for _, tl in self.mlp_loss_history]

                self.mlp_loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='MLP Error',
                )
                handles, _ = self.mlp_loss_ax.get_legend_handles_labels()
                if handles:
                    self.mlp_loss_ax.legend(loc='upper right')

            self.mlp_loss_ax.grid(True, alpha=0.3)
            self.mlp_loss_ax.set_title("MLP Training Loss")
            self.mlp_loss_ax.set_xlabel("Epoch")
            self.mlp_loss_ax.set_ylabel("Loss")

            self.mlp_loss_canvas.draw()
            self.mlp_loss_canvas.flush_events()
        
        else:
            self.loss_history.append((epoch, train_loss))
            self.loss_ax.clear()

            if len(self.loss_history) > 0:
                epochs = [e for e, _ in self.loss_history]
                train_losses = [tl for _, tl in self.loss_history]

                self.loss_ax.plot(
                    epochs,
                    train_losses,
                    'b-',
                    linewidth=2,
                    marker='o',
                    markersize=4,
                    label='Training Error',
                )
                handles, _ = self.loss_ax.get_legend_handles_labels()
                if handles:
                    self.loss_ax.legend(loc='upper right')

            self.loss_ax.grid(True, alpha=0.3)
            self.loss_ax.set_title("Training Loss")
            self.loss_ax.set_xlabel("Epoch")
            self.loss_ax.set_ylabel("Loss")

            self.loss_canvas.draw()

    def clear_loss_history(self, model_type=None):
        if model_type == 'Autoencoder' or model_type is None:
            self.ae_loss_history = []
            if hasattr(self, 'ae_loss_ax'):
                self.ae_loss_ax.clear()
                self.ae_loss_ax.set_xlabel('Epoch')
                self.ae_loss_ax.set_ylabel('Error')
                self.ae_loss_ax.set_title('Autoencoder Training Error')
                self.ae_loss_ax.grid(True, alpha=0.3)
                self.ae_loss_canvas.draw()
        
        if model_type in ['MLP', 'AutoencoderMLP'] or model_type is None:
            self.mlp_loss_history = []
            if hasattr(self, 'mlp_loss_ax'):
                self.mlp_loss_ax.clear()
                self.mlp_loss_ax.set_xlabel('Epoch')
                self.mlp_loss_ax.set_ylabel('Error')
                self.mlp_loss_ax.set_title('MLP Training Error')
                self.mlp_loss_ax.grid(True, alpha=0.3)
                self.mlp_loss_canvas.draw()
        
        if model_type is None or model_type not in ['Autoencoder', 'AutoencoderMLP']:
            self.loss_history = []
            if hasattr(self, 'loss_ax'):
                self.loss_ax.clear()
                self.loss_ax.set_xlabel('Epoch')
                self.loss_ax.set_ylabel('Error')
                self.loss_ax.set_title('Training Error')
                self.loss_ax.grid(True, alpha=0.3)
                self.loss_canvas.draw()

    def switch_to_tab(self, tab_name):
        tab_mapping = {
            'train': "🎯 Training",
            'test': "📊 Test",
            'loss': "📈 Error Graph",
            'ae_loss': "📈 Autoencoder Error",
            'mlp_loss': "📈 MLP Error",
        }
        if tab_name in tab_mapping:
            self.tabview.set(tab_mapping[tab_name])

    def enable_clicking(self, enabled=True):
        self.click_enabled = enabled
    
    def _add_tabs_for_mode(self, mode, model_type='MLP'):
        if mode == 'mnist':
            if model_type == 'AutoencoderMLP':
                self.tabview.add("📈 Autoencoder Error")
                self.tabview.add("📈 MLP Error")
                self.tabview.add("🔍 Reconstruction")
            else:
                self.tabview.add("📈 Error Graph")
        else:
            self.tabview.add("🎯 Training")
            self.tabview.add("📊 Test")
            self.tabview.add("📈 Error Graph")
    
    def configure_for_dataset_mode(self, mode, model_type='MLP'):
        if self.current_dataset_mode == mode and self.current_model_type == model_type:
            return
        
        self.current_dataset_mode = mode
        self.current_model_type = model_type
        
        try:
            current_tab = self.tabview.get()
        except:
            current_tab = None
        
        for tab_name in self.tabview._tab_dict.copy():
            self.tabview.delete(tab_name)
        
        self._add_tabs_for_mode(mode, model_type)
        
        if mode == 'manual':
            self._setup_train_tab()
            self._setup_test_tab()
            self._setup_loss_tab()
            self.tabview.set("🎯 Training")
        else:
            if model_type == 'AutoencoderMLP':
                self._setup_ae_loss_tab()
                self._setup_mlp_loss_tab()
                self._setup_reconstruction_tab()
                self.tabview.set("📈 Autoencoder Error")
            else:
                self._setup_loss_tab()
                self.tabview.set("📈 Error Graph")
            
            self.update()
            
            if model_type == 'AutoencoderMLP':
                if hasattr(self, 'ae_loss_canvas'):
                    try:
                        self.ae_loss_canvas.draw()
                        self.ae_loss_canvas.flush_events()
                    except:
                        pass
                if hasattr(self, 'mlp_loss_canvas'):
                    try:
                        self.mlp_loss_canvas.draw()
                        self.mlp_loss_canvas.flush_events()
                    except:
                        pass
            else:
                if hasattr(self, 'loss_canvas'):
                    try:
                        self.loss_canvas.draw()
                        self.loss_canvas.flush_events()
                    except:
                        pass
            
            def force_redraw_1():
                try:
                    self.update_idletasks()
                    if model_type == 'AutoencoderMLP':
                        if hasattr(self, 'ae_loss_canvas'):
                            self.ae_loss_canvas.draw_idle()
                            self.ae_loss_canvas.flush_events()
                        if hasattr(self, 'mlp_loss_canvas'):
                            self.mlp_loss_canvas.draw_idle()
                            self.mlp_loss_canvas.flush_events()
                        if hasattr(self, 'recon_canvas'):
                            self.recon_canvas.draw_idle()
                    else:
                        if hasattr(self, 'loss_canvas'):
                            self.loss_canvas.draw_idle()
                            self.loss_canvas.flush_events()
                except:
                    pass
            
            def force_redraw_2():
                try:
                    if model_type == 'AutoencoderMLP':
                        if hasattr(self, 'ae_loss_canvas'):
                            self.ae_loss_canvas.draw()
                            self.ae_loss_canvas.flush_events()
                        if hasattr(self, 'mlp_loss_canvas'):
                            self.mlp_loss_canvas.draw()
                            self.mlp_loss_canvas.flush_events()
                    else:
                        if hasattr(self, 'loss_canvas'):
                            self.loss_canvas.draw()
                            self.loss_canvas.flush_events()
                except:
                    pass
            
            self.after(200, force_redraw_1)
            self.after(500, force_redraw_2)
    
    def _setup_reconstruction_tab(self):
        tab = self.tabview.tab("🔍 Reconstruction")
        
        self.recon_fig = Figure(figsize=(7, 6), dpi=100)
        self.recon_ax = self.recon_fig.add_subplot(111)
        self.recon_ax.axis('off')
        self.recon_ax.set_title('Original vs Reconstructed Digits')
        
        self.recon_canvas = FigureCanvasTkAgg(self.recon_fig, tab)
        self.recon_canvas.draw()
        self.recon_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def update_reconstruction(self, original_images, reconstructed_images, mse_per_sample=None):
        self.recon_ax.clear()
        self.recon_ax.axis('off')
        
        n_samples = min(len(original_images), 10)
        if n_samples == 0:
            self.recon_canvas.draw()
            return
        
        self.recon_fig.clear()
        
        # Use 4 rows x 5 columns: 
        # Row 0-1: Original images (2 rows for labels)
        # Row 2-3: Reconstructed images (2 rows for labels)
        n_cols = 5
        
        for i in range(n_samples):
            # Original images (top half)
            row = i // n_cols
            col = i % n_cols
            
            ax_orig = self.recon_fig.add_subplot(4, n_cols, row * n_cols + col + 1)
            img_orig = [original_images[i][j:j+28] for j in range(0, 784, 28)]
            ax_orig.imshow(img_orig, cmap='gray')
            ax_orig.axis('off')
            if i == 0:
                ax_orig.set_title('Original', fontsize=10)
            
            # Reconstructed images (bottom half)
            ax_recon = self.recon_fig.add_subplot(4, n_cols, (row + 2) * n_cols + col + 1)
            img_recon = [reconstructed_images[i][j:j+28] for j in range(0, 784, 28)]
            ax_recon.imshow(img_recon, cmap='gray')
            ax_recon.axis('off')
            
            if i == 0:
                ax_recon.set_title('Reconstructed', fontsize=10)
            
            if mse_per_sample is not None and i < len(mse_per_sample):
                ax_recon.text(0.5, -0.1, f'MSE: {mse_per_sample[i]:.4f}', 
                             ha='center', va='top', transform=ax_recon.transAxes, fontsize=8)
        
        self.recon_fig.tight_layout()
        self.recon_canvas.draw()
