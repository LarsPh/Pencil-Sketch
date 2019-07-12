# Pencil-Sketch
Produce pencil drawing from natural images.

## About the algorithm
* The algorithm is mainly based on "Combining Sketch and Tone for Pencil Drawing Production" Cewu Lu, Li Xu, Jiaya Jia, International Symposium on Non-Photorealistic Animation and Rendering (NPAR 2012), June, 2012 (http://www.cse.cuhk.edu.hk/~leojia/projects/pencilsketch/pencil_drawing.htm)

* In the edge detection step, instead of the gradient method provided in the paper, bilateral filter and Laplacian edge detection are used to generate a less noisy image, which significantly impact the quality of final result in this implementation.

## Other references
* https://blog.csdn.net/qibofang/article/details/51482431
* https://www.cnblogs.com/Imageshop/p/4285566.html
* https://blog.csdn.net/bluecol/article/details/45422763

## Getting Started

### Prerequisites
Modules required: numpy, scipy, Pillow, opencv-python, matplotlib.

### Running the program
Call function pencilSketch() to run the program, you can also specify the parameters to change the looking of out image. For details of parameters please check the comment in source code and the references.

### Result images
The default directory program looks for input and texture images are "soure_images" and "textures", you can modify the folders' name by specifying them when calling the function. The output colored and grayscale pencil sketch images are saved in folder output. The intermediate images are saved in folder temp.

## Gallery
![test1](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/input/test1.jpg)
![test1_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/pencil_sketch_test1.jpg)
![test1_colored_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/colored_pencil_sketch_test1.jpg)
![test2](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/input/test2.png)
![test2_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/pencil_sketch_test2.png)
![test2_colored_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/colored_pencil_sketch_test2.png)
![test3](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/input/test3.jpg)
![test3_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/pencil_sketch_test3.jpg)
![test3_colored_output](https://raw.githubusercontent.com/LarsPh/Pencil-Sketch/master/gallery/output/colored_pencil_sketch_test3.jpg)
