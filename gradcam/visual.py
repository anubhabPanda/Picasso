import cv2
import torch

import matplotlib.pyplot as plt
import numpy as np
from .gradcam import GradCAM
from .gradcam_pp import GradCAMPP
from utils import to_numpy, unnormalize


def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.

    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Returns:

        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


class GradCAMView:

    def __init__(self, model, layers, device, mean, std):
        """Instantiate GradCAM and GradCAM++.

        Args:
            model: Trained model.
            layers: List of layers to show GradCAM on.
            device: GPU or CPU.
            mean: Mean of the dataset.
            std: Standard Deviation of the dataset.
        """
        self.model = model
        self.layers = layers
        self.device = device
        self.mean = mean
        self.std = std

        self._gradcam()
        self._gradcam_pp()

        print('Mode set to GradCAM.')
        self.grad = self.gradcam.copy()

        self.views = []

    def _gradcam(self):
        """ Initialize GradCAM instance. """
        self.gradcam = {}
        for layer in self.layers:
            self.gradcam[layer] = GradCAM(self.model, layer)
    
    def _gradcam_pp(self):
        """ Initialize GradCAM++ instance. """
        self.gradcam_pp = {}
        for layer in self.layers:
            self.gradcam_pp[layer] = GradCAMPP(self.model, layer)
    
    def switch_mode(self):
        if self.grad == self.gradcam:
            print('Mode switched to GradCAM++.')
            self.grad = self.gradcam_pp.copy()
        else:
            print('Mode switched to GradCAM.')
            self.grad = self.gradcam.copy()
    
    def _cam_image(self, norm_image):
        """Get CAM for an image.

        Args:
            norm_image: Normalized image. Should be of type
                torch.Tensor
        
        Returns:
            Dictionary containing unnormalized image, heatmap and CAM result.
        """
        norm_image_cuda = norm_image.clone().unsqueeze_(0).to(self.device)
        heatmap, result = {}, {}
        for layer, gc in self.gradcam.items():
            mask, _ = gc(norm_image_cuda)
            cam_heatmap, cam_result = visualize_cam(
                mask,
                unnormalize(norm_image, self.mean, self.std, out_type='tensor').clone().unsqueeze_(0).to(self.device),
                alpha = 0.6
            )
            heatmap[layer], result[layer] = to_numpy(cam_heatmap), to_numpy(cam_result)
        return {
            'image': unnormalize(norm_image, self.mean, self.std),
            'heatmap': heatmap,
            'result': result
        }
    
    def _plot_view(self, view, fig, row_num, ncols, metric):
        """Plot a CAM view.

        Args:
            view: Dictionary containing image, heatmap and result.
            fig: Matplotlib figure instance.
            row_num: Row number of the subplot.
            ncols: Total number of columns in the subplot.
            metric: Can be one of ['heatmap', 'result'].
        """
        sub = fig.add_subplot(row_num, ncols, 1)
        sub.axis('off')
        plt.imshow(view['image'])
        sub.set_title(f'{metric.title()}:')
        for idx, layer in enumerate(self.layers):
            sub = fig.add_subplot(row_num, ncols, idx + 2)
            sub.axis('off')
            plt.imshow(view[metric][layer])
            sub.set_title(layer)
    
    def cam(self, norm_image_list):
        """Get CAM for a list of images.

        Args:
            norm_image_list: List of normalized images. Each image
                should be of type torch.Tensor
        """
        for norm_image in norm_image_list:
            self.views.append(self._cam_image(norm_image))
    
    def plot(self, plot_path, plot_size):
        """Plot heatmap and CAM result.

        Args:
            plot_path: Path to save the plot.
        """
    
        for idx, layer in enumerate(self.layers):
            
            n_images = len(self.views)
            n_cols = int(np.sqrt(n_images))
            n_rows = int(np.ceil(np.sqrt(n_images)))
            # Initialize plot    
            fig, axes = plt.subplots(n_rows, n_cols, figsize = plot_size)
            axises = axes.flatten() if n_images > 1 else [axes]
            for i, ax in enumerate(axises):
                ax.axis('off')
                if i < n_images:
                    grad_cam_image = np.clip(self.views[i]['result'][layer], 0, 1)
                    ax.imshow(grad_cam_image)   
                
            # Set spacing and display
            plt.subplots_adjust(wspace=0, hspace=0)
            fig.suptitle(f"Gradcams for {layer.upper()}", fontsize = 15, y=1.05)
            fig.tight_layout()
            plt.show()

            # Save image
            fig.savefig(f'{plot_path}_{idx + 1}.png', bbox_inches='tight')

            # Clear cache
            plt.clf()
    
    def __call__(self, norm_image_list, plot_path, plot_size = (15, 15)):
        """Get GradCAM for a list of images.

        Args:
            norm_image_list: List of normalized images. Each image
                should be of type torch.Tensor
        """
        self.cam(norm_image_list)
        self.plot(plot_path, plot_size)
