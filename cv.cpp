#include <iostream>
#include"stdafx.h"

using namespace cv;
using namespace std;

Mat wiener2(Mat I, Mat image_spectrum, int noise_stddev);
Mat padd_image(Mat I);

Mat get_spectrum(Mat I);
Mat get_dft(Mat I);

Mat with_noise(Mat image, int stddev);
Mat rand_noise(Mat I, int stddev);

Mat createavg(Size imsize);
void shift(Mat magI);


int main(int argc, char* argv[]) {

    int noise_stddev = 20;
    string input_filename = "blur.png", output_filename = "write.png";   // Have a blurred image here
    cout << "noise standard deviation: " << noise_stddev << "\n";
    cout << "input file: " << input_filename << "\n";

    Mat I = imread(input_filename, IMREAD_GRAYSCALE);
    if (I.data == NULL) {
        cout << "Can't open file: " << input_filename << "\n";
        return 2;
    }

    Mat raw_sample = imread("blur.png", IMREAD_GRAYSCALE);
    if (raw_sample.data == NULL) {
        cout << "Can't open file: sample.bmp\n";
        return 3;
    }

    Mat padded = padd_image(I);
    Mat noisy;

    noisy = with_noise(padded, noise_stddev);

    Mat sample(padded.rows, padded.cols, CV_8U);
    resize(raw_sample, sample, sample.size());
    Mat spectrum = get_spectrum(sample);    //to get signal spectrum of known image 
    Mat enhanced = wiener2(noisy, spectrum, noise_stddev);
    imshow("image 1", noisy);
    imshow("image 2", enhanced);
    waitKey();
}
Mat createavg(Size imsize) {


    Mat kernel = Mat(5, 5, CV_32FC1, Scalar(0.04));

    int w = imsize.width - kernel.cols;
    int h = imsize.height - kernel.rows;

    int r = w / 2;
    int l = imsize.width - kernel.cols - r;

    int b = h / 2;
    int t = imsize.height - kernel.rows - b;

    Mat ret;
    copyMakeBorder(kernel, ret, t, b, l, r, BORDER_CONSTANT, Scalar::all(0));

    return ret;

}

//inputs are the blurry image with noise , the original image power spectra , and standard deviation of the noise introduced
Mat wiener2(Mat final_noise, Mat image_spectrum, int noise_stddev) {
    Mat padded = padd_image(final_noise);
    Mat noise = rand_noise(padded, noise_stddev);
    Mat noise_spectrum = get_spectrum(noise);

    Scalar padded_mean = mean(padded);

    Mat planes[2];
    Mat complexI = get_dft(padded);
    split(complexI, planes);    // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))

    Mat factor = (noise_spectrum / image_spectrum); //calculates the signal to noise ratio
    //-----------------compute the frequency domain multiplier



    Mat mask = createavg(padded.size());            //creating the kernel which initally prduced the blurred image
    shift(mask);// shifting the filter
    Mat mplane[] = { Mat_<float>(mask), Mat::zeros(mask.size(), CV_32F) };
    Mat kernelcomplex;
    merge(mplane, 2, kernelcomplex);

    dft(kernelcomplex, kernelcomplex);  // computing dft of kernel

    split(kernelcomplex, mplane);// splitting the dft of kernel to real and complex 
    Mat x = mplane[0];
    //cout<<x;

    magnitude(mplane[0], mplane[1], mplane[0]);// planes[0] = magnitude
    Mat magI = mplane[0];
    //cout<<magI;
    multiply(magI, magI, magI);        //Computing |H(f)|^2
    //cout<<factor;
    factor += magI;                   //adding to signal to noise ratio
    //cout<<factor;
    magI = magI / factor;   // calculating  (|H(f)|^2)/(|H(f)|^2 + S/N)         

    //cout<<magI << " "<<x;
    magI = magI / x;                //Dividing by the real value part of dft of kernel thus effectively multiplying by (1/H(f))
    factor = magI;
    //cout<<magI;

    //-------------------end


    //multply both frequency domains and get final image in frequency domain
    multiply(planes[0], factor, planes[0]);
    multiply(planes[1], factor, planes[1]);


    merge(planes, 2, complexI);
    idft(complexI, complexI);
    split(complexI, planes);
    //  normalize(planes[0], planes[0], 0, 128, CV_MINMAX );
    Scalar enhanced_mean = mean(planes[0]);
    double norm_factor = padded_mean.val[0] / enhanced_mean.val[0];
    multiply(planes[0], norm_factor, planes[0]);
    Mat normalized;
    planes[0].convertTo(normalized, CV_8UC1);
    return normalized;
}

Mat padd_image(Mat I) {
    Mat padded;
    int m = getOptimalDFTSize(I.rows);
    int n = getOptimalDFTSize(I.cols); // on the border add zero pixels
    copyMakeBorder(I, padded, 0, m - I.rows, 0, n - I.cols, BORDER_CONSTANT, Scalar::all(0));
    return padded;
}

Mat get_spectrum(Mat I) {
    Mat complexI = get_dft(I);
    Mat planes[2];
    split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
    magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
    Mat magI = planes[0];
    multiply(magI, magI, magI);
    return magI;
}

Mat get_dft(Mat I) {
    Mat image;
    I.convertTo(image, CV_32F);
    Mat planes[] = { Mat_<float>(image), Mat::zeros(image.size(), CV_32F) };
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    return complexI;
}

Mat with_noise(Mat image, int stddev) {
    Mat noise(image.rows, image.cols, CV_8U);
    rand_noise(image, stddev).convertTo(noise, CV_8U);
    Mat noisy = image.clone();
    noisy += noise;
    return noisy;
}

Mat rand_noise(Mat I, int stddev) {
    Mat noise = Mat::zeros(I.rows, I.cols, CV_32F);
    randn(noise, Scalar::all(0), Scalar::all(stddev));
    return noise;
}
void shift(Mat magI) {

    // crop if it has an odd number of rows or columns
    magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

    int cx = magI.cols / 2;
    int cy = magI.rows / 2;

    Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                            // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);                     // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}