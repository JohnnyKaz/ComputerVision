"""
CS-472: Computer Vision (Spring Semester 2023)
Assignment 5: Particle Swarm Optimization Tracking
Date: 03/07/2023
University of Crete, Department of Computer Science

By: Ioannis (Yannis) Kaziales ~ csdp1305

This file contains the implementation of the following classes:
    - Particle:                 represents a single particle of the swarm
    - ParticleSwarmOptimizer:   implements the Particle Swarm Optimization algorithm
as well as some utility functions for training, plotting and evaluation:
    - initialize_particles: initializes the particles of the swarm
    - mIoU:                 calculates the mean Intersection over Union (mIoU) of a circle fitted on an image
    - evaluate_fitness:     evaluates the fitness of a particle's position (i.e. the loss of the circle fitted on the image: 1 - mIoU)
    - plot_iteration:       plots the current iteration of the PSO algorithm (i.e. all the particles and their circles in the image)
    - plot_estimation:      plots a single circle/ellipse (the one fitted using PSO) on an image
    - plot_loss:            plots the loss curve of the execution of the PSO algorithm
    - make_video:           creates a video from a directory containing images
    - preprocess_image:     preprocesses an image, i.e. binarizes it and applies morphological operations to reduce noise
    - save_txt:             saves the estimated parameters to a .txt file as per the required format
"""
import numpy as np
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class Particle:
    """
    Represents a single particle of the swarm
    Attributes: position, velocity, best position and best fitness
    """
    def __init__(self, position:list, velocity:list=None):
        self.position = position
        self.velocity = velocity if velocity is not None else [0]*len(position)
        self.best_position = position
        self.best_fitness = float('inf')


class ParticleSwarmOptimizer():
    """
    Implements the Particle Swarm Optimization algorithm
    Methods:
        - optimize:     runs the PSO algorithm
    """
    def __init__(self, num_particles:int, max_iterations:int, c1:float, c2:float, w:float, decay:float=0.99,
                 stop_iterations:int=10, global_best_position:list=None, global_best_fitness:float=float('inf')):
        """
        Initializes the Particle Swarm Optimizater and its parameters
        Input arguments:
            - num_particles:        the number of particles of the swarm
            - max_iterations:       the maximum number of iterations to run the algorithm for
            - c1:                   the cognitive parameter (defines how much the individual's best position affects its velocity)
            - c2:                   the social parameter (defines how much the global best position affects the individual's velocity)
            - w:                    the inertia parameter (defines how much the previous velocity affects the current velocity)
            - decay:                the decay parameter (defines how much the inertia parameter decays after each iteration) (default: 0.99)
            - stop_iterations:      the number of iterations to stop the algorithm if no improvement is made (default: 10)
            - global_best_position: the global best position (default: None - we have no prior knowledge of the best position)
            - global_best_fitness:  the global best fitness (default: float('inf') - we have no prior knowledge of the best fitness)
        """
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.decay = decay
        self.stop_iterations = stop_iterations
        self.losses = []
        self.global_best_position = global_best_position
        self.global_best_fitness = global_best_fitness

    def optimize(self, img:np.ndarray, particles:list, max_position:list):
        """
        Runs the PSO algorithm
        Input arguments:
            - img:          the image to fit the circle/ellipse on
            - particles:    the particles of the swarm
            - max_position: the maximum position of a particle (i.e. the maximum x, y and r values for a circle)
        Output arguments:
            - global_best_position: the global best position (i.e. the best circle/ellipse fitted on the image)
            - global_best_fitness:  the global best fitness (i.e. the loss of the best circle/ellipse fitted on the image)
            - losses:               the losses of the best circle/ellipse fitted on the image for each iteration
        """
        iters_no_improvement = 0
        for iter in range(self.max_iterations):
            iters_no_improvement += 1
            for particle in particles:
                # Evaluate fitness of particle's position
                fitness = evaluate_fitness(img, particle.position)
                # Update particle's best-known position
                if fitness < particle.best_fitness:
                    particle.best_fitness = fitness
                    particle.best_position = particle.position
                # Update global best position
                if fitness < self.global_best_fitness:
                    self.global_best_fitness = fitness
                    self.global_best_position = particle.position
                    iters_no_improvement = 0
                # Update particle's velocity
                particle.velocity = self.w * particle.velocity + \
                                    self.c1 * np.random.rand() * (particle.best_position - particle.position) + \
                                    self.c2 * np.random.rand() * (self.global_best_position - particle.position)
                # Update particle's position
                particle.position = particle.position + particle.velocity
                # Keep particle inside image
                particle.position = np.clip(particle.position, 0, max_position).astype(int)
            self.w = self.w * self.decay
            self.losses.append(self.global_best_fitness)
            if iters_no_improvement >= self.stop_iterations:
                # print(f"Stopping after {iter+1} iterations with no improvement")
                break
        return self.global_best_position, self.global_best_fitness, self.losses


def initialize_particles(image:np.ndarray, num_particles:int, max_position:list, use_floats:bool=False):
    """
    Initializes the particles of the swarm
    Input:
        - image:         the image to fit the circle/ellipse on
        - num_particles: the number of particles of the swarm
        - max_position:  the maximum position of a particle in a list (i.e. the maximum x, y and r values for a circle)
        - use_floats:    whether to use floats or integers for the position of the particles (default: integers)
    Output:
        - particles:     the particles of the swarm (a list of Particle objects)
    """
    particles = []
    rand = np.random.uniform if use_floats else np.random.randint
    for _ in range(num_particles):
        position = rand(low=[0]*len(max_position), high=max_position)
        velocity = np.zeros(len(max_position))
        particle = Particle(position, velocity)
        particles.append(particle)
    return particles

def mIoU(img:np.ndarray, position:list):
    """
    Computes the mean Intersection over Union (mIoU) between the image and the circle/ellipse defined by the position
    Input:
        - img:      the image to fit the circle/ellipse on
        - position: the position of the circle/ellipse (a list of 3 (for circle) or 5 elements (for ellipse))
    Output:
        - mIoU:     the mean Intersection over Union (mIoU) between the image and the circle/ellipse defined by the position
    """
    mask = np.zeros_like(img)
    if len(position) == 3:
        # Circle
        x, y, r = position
        cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    elif len(position) == 5:
        # Ellipse
        x, y, a, b, theta = position
        cv2.ellipse(mask, (int(x), int(y)), (int(a), int(b)), int(theta), 0, 360, 255, -1)
    else:
        raise ValueError("Invalid position: must be a list of 3 (for circle) or 5 elements (for ellipse)")
    interscection = np.bitwise_and(mask, img)
    union = np.bitwise_or(mask, img)
    return np.sum(interscection) / np.sum(union)

def evaluate_fitness(img:np.ndarray, particle_pos:list):
    """
    Evaluates the fitness of the circle/ellipse defined by the position
    Input:
        - img:          the image to fit the circle/ellipse on
        - particle_pos: the position of the circle/ellipse (a list of 3 (for circle) or 5 elements (for ellipse))
    Output:
        - fitness:      the fitness/loss of the circle/ellipse defined by the position
    """
    return 1 - mIoU(img, particle_pos)

def plot_iteration(img:np.ndarray, particles:list, colors:list, title:str, show:bool=True, save_filename:str=None):
    """
    Plots the image with all the particles of the swarm overlaid
    Works for circles only
    Input:
        - img:           the image to fit the circle/ellipse on
        - particles:     the particles of the swarm (a list of Particle objects)
        - colors:        the colors of the particles
        - title:         the title of the plot
        - show:          whether to show the plot (default: True)
        - save_filename: the filename to save the plot (default: None -> don't save)
    """
    y_max, x_max = img.shape[:2]
    plt.figure()
    plt.imshow(img, cmap='gray')
    for i, particle in enumerate(particles):
        x, y, r = particle.position
        plt.scatter(x, y, color=colors[i], marker='x')
        plt.gca().add_patch(plt.Circle((x, y), r, color=colors[i], alpha=0.2))
    plt.title(title)
    plt.xlim(0, x_max)
    plt.ylim(y_max, 0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)
    if show:
        plt.show()
    else:
        plt.close()

def plot_estimation(img:np.ndarray, params:list, title:str, save_filename:str=None, est_type:str='circle', show:bool=True, rgb:bool=False):
    """
    Plots the image with the estimated circle/ellipse overlaid
    Input:
        - img:           the image to fit the circle/ellipse on
        - params:        the parameters of the estimated circle/ellipse (a list of 3 (for circle) or 5 elements (for ellipse))
        - title:         the title of the plot
        - save_filename: the filename to save the plot (default: None -> don't save)
        - est_type:      the type of the estimated shape (default: 'circle', available: 'circle', 'ellipse')
        - show:          whether to show the plot (default: True)
        - rgb:           whether the image is in RGB format or grayscale (default: False -> grayscale)
    """
    y_max, x_max = img.shape[:2]
    plt.figure()
    plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) if rgb else plt.imshow(img, cmap='gray')
    if est_type == 'circle':
        x, y, r = params
        plt.gca().add_patch(plt.Circle((x, y), r, color='red', fill=False, linewidth=1))
    elif est_type == 'ellipse':
        x, y, a, b, theta = params
        ellipse = Ellipse((x, y), 2*a, 2*b, theta, color='red', fill=False, linewidth=1)
        plt.gca().add_patch(ellipse)
    else:
        raise ValueError(f"Unknown type '{type}'. Accepted types are 'circle' and 'ellipse'")
    plt.scatter(x, y, color='red', marker='x')
    # show in the range [0, x_max], [0, y_max], with the origin at the top left corner
    plt.xlim(0, x_max)
    plt.ylim(y_max, 0)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename, bbox_inches='tight', dpi=196)
    if show:
        plt.show()
    else:
        plt.close()

def plot_loss(losses:list, title:str, max_iterations:int=100, save_filename:str=None):
    """
    Plots the loss (evaluation function result) over the iterations
    Input:
        - losses:         the losses over the iterations
        - title:          the title of the plot
        - max_iterations: the maximum number of iterations to show in the x-axis (default: 100)
        - save_filename:  the filename to save the plot (default: None -> don't save)
    """
    plt.figure()
    plt.plot(losses, color='red', linewidth=2)
    plt.title(title)
    plt.xlabel('Iteration')
    plt.ylabel('Loss (1 - mIoU)')
    plt.xlim(0, max_iterations)
    plt.xticks(np.arange(0, max_iterations+1, 10))
    plt.tight_layout()
    if save_filename:
        plt.savefig(save_filename)
    plt.show()

def make_video(input_dir:str, filename_prefix:str, width:int=640, height:int=480, fps:int=10, 
               extension:str='png', num_images:int=None, num_pad:int=0, savepath:str='video.avi'):
    """
    creates a video from a sequence of images contained in a directory.
    The images must be in the format 'filename_prefix_0.png', 'filename_prefix_1.png', ...
    Inputs:
        - input_dir:            str     directory containing the images
        - filename_prefix:      str     prefix of the image filenames (e.g. 'image' for 'image_0.png', 'image_1.png', ...)
        - width:                int     width of the video (default: 640)
        - height:               int     height of the video (default: 480)
        - fps:                  int     frames per second (default: 10)
        - extension:            str     extension of the image files (default: 'png')
        - num_images:           int     number of images to use (default: all images in the directory)
        - num_pad:              int     number of zeros used to pad the image number (default: no padding)
        - savepath:             str     path where to save the video (default: 'video.avi')
    """
    if num_images is None:
        num_images = len([name for name in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, name))])
    fourcc = cv2.VideoWriter_fourcc(*"mp4v") 
    video = cv2.VideoWriter(savepath, fourcc, fps, (width, height))
    for i in tqdm(range(num_images)):
        filename = f"{filename_prefix}_{str(i).zfill(num_pad)}.{extension}"
        img = cv2.imread(os.path.join(input_dir, filename))
        video.write(img)
    cv2.destroyAllWindows()
    video.release()
    print(f"Video '{savepath}' saved")

def preprocess_image(img:np.ndarray, save_filename:str=None, threshold:int=190, open_size:int=110, close_size:int=5):
    """
    Preprocesses a grayscale image to extract the circle while removing the background, using morphological operations.
    Inputs:
        - img:           ndarray        image to preprocess
        - save_filename: str            filename to save the masked image (default: None -> don't save)
        - threshold:     int            threshold to apply to the image for binarization (default: 190)
        - open_size:     int            size of the kernel to use for the opening (default: 110)
        - close_size:    int            size of the kernel to use for the closing (default: 5)
    Outputs:
        - masked:        np.ndarray     final binary image with the circle masked
    """
    kernel_close = np.ones((close_size, close_size), np.uint8)
    kernel_open = np.ones((open_size, open_size), np.uint8)
    # apply threshold to get binary image
    thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    # apply closing to fill the circle
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_close)
    # apply opening with big kernel to remove background noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    # apply dilation to make the circle bigger (encompass the whole circle)
    mask = cv2.dilate(opened, kernel_open, iterations=1)
    # apply mask to original image
    masked = cv2.bitwise_and(closed, closed, mask=mask)
    # save masked image
    if save_filename:
        cv2.imwrite(save_filename, masked)
    #cv2.imwrite(os.path.join(out_dir, filename.split('.')[0]+'.png'), masked)
    return masked

def preprocess_image_ellipse(img:np.ndarray, save_filename:str=None, threshold:int=190, open_size:int=150, close_size:int=70):
    dilation_size = round(0.5 * open_size)
    kernel_close  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    kernel_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_size, open_size))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_size, dilation_size))
    # get binary image
    img_adaptive = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 501, -25)
    thresh = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)[1]
    thresholded = cv2.bitwise_and(img_adaptive, thresh)
    # apply closing to fill the circle
    closed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel_close)
    # apply opening with big kernel to remove background noise
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    # apply dilation to make the circle bigger (encompass the whole circle)
    mask = cv2.dilate(opened, kernel_dilate, iterations=1)
    # apply mask to original image
    masked = cv2.bitwise_and(thresholded, thresholded, mask=mask)
    # save masked image
    if save_filename:
        cv2.imwrite(save_filename, masked)
    return masked


def save_txt(filename:str, estimated_params:list, type:str='circle'):
    """
    Saves the estimated parameters in a text file
    Inputs:
        - filename:          str     path where to save the text file
        - estimated_params:  list    the estimated parameters (i.e. [x, y, r] for a circle or [x, y, a, b, theta] for an ellipse)
        - type:              str     the type of the estimated shape (default: 'circle', accepted values: 'circle', 'ellipse')
    """
    with open(filename, 'w') as f:
        if type == 'circle':
            x, y, r = estimated_params
            f.write(f"Center: {x}, {y}\nRadius: {r}")
        elif type == 'ellipse':
            x, y, a, b, theta = estimated_params
            f.write(f"Center: {x}, {y}\nMajor axis: {a}\nMinor axis: {b}\nAngle: {theta}")
        else:
            raise ValueError(f"Unknown type '{type}'. Accepted types are 'circle' and 'ellipse'")