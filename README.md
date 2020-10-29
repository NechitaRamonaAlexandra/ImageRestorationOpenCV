# ImageRestorationOpenCV

In this project I worked with OpenCV and C++ in order to design and comprehend complex image processing algorithms. The main highlight of the project is the implementation of the **Wiener Filter**.

---
## Image Restoration
Image restoration is performed by reversing the process that blurred the image and such is performed by imaging a point source and use the point source image, which is called the Point Spread Function (PSF) to restore the image information lost to the blurring process.  
The Wiener filter is one of the most robust filters for solving problems of this kind, restoring signals in the presence of additive noise. It can be used with data of single or dual polarity and for 1D or 2D signal processing problems which are the result of linear time invariant processes and non-causal.
It uses deconvolution and Fourier transformations to restore some of the initial qualities of the image.

---
## Implementation
I worked on grayscale images, for convenience. I took a regular image and I added some artificial blur noise, using the `randn()` function with some default value. The main focus of the algorithm were Fourier transformations and used the `dft()` function. I computed the equation from the definition of the filter and transformed the image back from the frequency domain, obtaining a slightly blurred but smoother image, with no noise.

The most important function in the project  (being the one that implements this filter) is the function `wienerFilter()`, which can be executed by introducing 100 in stdin of the application.

## Obtained Results
![Image](https://github.com/NechitaRamonaAlexandra/ImageRestorationOpenCV/blob/main/output.png)
