// created by Yang Zhou, 11/11/2015

#include <opencv2/opencv.hpp>
#include <math.h>
#include <assert.h>
#include <iostream>
#include <fstream>
#include <iomanip>
#include "argvparser.h"
using namespace std;
using namespace cv;
using namespace CommandLineProcessing;
 
// The VLFeat header files need to be declared external.
extern "C" {
    #include <vl/generic.h>
    #include <vl/covdet.h>
    #include <vl/mathop.h>
    #include <vl/sift.h>
}

void flip_descriptor (float *dst, float const *src)
{
    int const BO = 8;  /* number of orientation bins */
    int const BP = 4;  /* number of spatial bins     */
    int i, j, t;

    for (j = 0; j < BP; ++j) {
        int jp = BP - 1 - j;
        for (i = 0; i < BP; ++i) {
            int o  = BO * i + BP*BO * j;
            int op = BO * i + BP*BO * jp;
            dst [op] = src[o];
            for (t = 1; t < BO; ++t) {
                dst [BO - t + op] = src [t + o];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    /* define some public options */
    vl_bool estimateAffineShape = VL_TRUE;
    vl_bool estimateOrientation = VL_TRUE;
    // vl_bool doubleImage = VL_FALSE;
    vl_bool doubleImage = VL_FALSE;
    double peakThreshold = 0.001; 
    double edgeThreshold = -1;
    double lapPeakThreshold = -1;
    vl_bool verbose = VL_FALSE;
    
    /* read in command line options */
    ArgvParser cmd; 
    cmd.setIntroductoryDescription("Hessian affine feature detector + sift descriptor.");

    cmd.addErrorCode(0, "Success");
    cmd.addErrorCode(-1, "Error");
    cmd.setHelpOption("h", "help", "useage: ./sift -estimateaffineshape -estimateorientation -peakthreshold=0.001 imgpath outpath");

    cmd.defineOption("doubleimg", "bool option: increase image resolution");
    cmd.defineOption("estimateaffineshape", "bool option: estimate affineshape");
    cmd.defineOption("estimateorientation", "bool option: estimate orientation");
    cmd.defineOption("verbose", "bool option: output more debug info");
    cmd.defineOption("peakthreshold", "peak threhold value for hessian, default value is 0.001", ArgvParser::OptionRequiresValue);

    int result = cmd.parse(argc, argv);
    if (result != ArgvParser::NoParserError || cmd.arguments() != 2) {
        if (cmd.arguments() != 2) {
            cerr << cmd.parseErrorDescription(result) << endl;
        }
        else {
            cerr << "Only two arguments is allowed for image path" << endl;
        }
        exit(-1);
    }

    string imgPath = cmd.argument(0);
    string outPath = cmd.argument(1);
    if (cmd.foundOption("doubleimg")) {
        doubleImage = VL_FALSE;
    }
    if (cmd.foundOption("estimateaffineshape")) {
        estimateAffineShape = VL_TRUE;    
    }
    if (cmd.foundOption("estimateorientation")) {
        estimateOrientation = VL_TRUE;    
    }
    if (cmd.foundOption("peakthreshold")) {
        peakThreshold = stod(cmd.optionValue("peakthreshold"));
    }
    if (cmd.foundOption("verbose")) {
        verbose = VL_TRUE;    
    }

    /* read image file name ,and convert to 1-d image */
    cout << "process " << imgPath << "... ..." << endl;
    Mat gray = cv::imread(imgPath, 0);
    if (!gray.data) {
        cerr << "No image data" << endl;
        exit(-1);
    }

    // store image by row
    int rows = gray.rows, cols = gray.cols;
    // float* image = new float[rows * cols]();
    vector<float> nums;
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            // image[j * rows + i] = gray.at<uchar>(i, j);
            nums.push_back(((float) gray.at<uchar>(i, j))/255.0);
        }
    }
    float *image = &nums[0]; 

    /* define some private options */
    VlCovDetMethod method = VL_COVDET_METHOD_HESSIAN_LAPLACE;
    // VlCovDetMethod method = VL_COVDET_METHOD_HESSIAN;
    // VlCovDetMethod method = VL_COVDET_METHOD_DOG;
    vl_index octaveResolution = -1;
    double boundaryMargin = 2.0;
    vl_index patchResolution = -1;
    double patchRelativeExtent = -1;
    double patchRelativeSmoothing = -1;
    float *patch = NULL;
    float *patchXY = NULL;
    if (patchResolution < 0)  patchResolution = 15;
    if (patchRelativeExtent < 0) patchRelativeExtent = 7.5;
    if (patchRelativeSmoothing < 0) patchRelativeSmoothing = 1;

    if (patchResolution > 0) {
        vl_size w = 2 * patchResolution + 1;
        patch = (float *)malloc(sizeof(float) * w * w);
        patchXY = (float *)malloc(2 * sizeof(float) * w * w);
    }


    VlCovDet * covdet = vl_covdet_new(method);

    vl_covdet_set_transposed(covdet, VL_TRUE) ;
    /* set covdet parameters */
    vl_covdet_set_first_octave(covdet, doubleImage ? -1 : 0);
    if (octaveResolution >= 0) vl_covdet_set_octave_resolution(covdet, octaveResolution);
    if (peakThreshold >= 0) vl_covdet_set_peak_threshold(covdet, peakThreshold);
    if (edgeThreshold >= 0) vl_covdet_set_edge_threshold(covdet, edgeThreshold);
    if (lapPeakThreshold >= 0) vl_covdet_set_laplacian_peak_threshold(covdet, lapPeakThreshold);
    
    if (verbose) {
        VL_PRINTF("vl_covdet: doubling image: %s\n",
                VL_YESNO(vl_covdet_get_first_octave(covdet) < 0)) ;

        cout << "vl_covdet: detector: "  
             << vl_enumeration_get_by_value(vlCovdetMethods, method)->name
             << endl;

        cout << "vl_covdet: peak threshold: " 
             << vl_covdet_get_peak_threshold(covdet)
             << "  edge threshold: "
             << vl_covdet_get_edge_threshold(covdet)
             << endl;
    }
    /* process the image */
    vl_covdet_put_image(covdet, image, rows, cols);
    vl_covdet_detect(covdet);

    vl_size numFeatures = vl_covdet_get_num_features(covdet);
    if (verbose) {
        cout << "vl_covdet: " <<  vl_covdet_get_num_non_extrema_suppressed(covdet) << 
         " features suppressed as duplicate (threshold: " << 
         vl_covdet_get_non_extrema_suppression_threshold(covdet) << ")" << endl;
        VL_PRINTF("vl_covdet: detected %d features\n", numFeatures) ;
    }
    // ***************************************************************************************


    if (verbose) {
        vl_size numScales;
        vl_size const * numFeaturesPerScale;
        numFeaturesPerScale = vl_covdet_get_laplacian_scales_statistics(covdet, &numScales);
        cout << "vl_covdet: Laplacian scales:";
        for (int k = 0; k <= (signed)numScales; ++k) {
            char message[128];
            sprintf(message, "%d with %d scales;", (int) numFeaturesPerScale[k], k);
            cout << message;
        }
        cout << endl;
    }

    /* boundary setting */
    if (boundaryMargin > 0) {
        vl_covdet_drop_features_outside(covdet, boundaryMargin);
    }

    /* affine adaptation if needed */
    if (estimateAffineShape) {
        vl_covdet_extract_affine_shape(covdet);
    }

    /* orientation estimation if needed */
    if (estimateOrientation) {
        vl_covdet_extract_orientations(covdet);
    }

    /* extract sift descriptor */
    /* open a outPath for sift descriptors*/
    ofstream ofs(outPath);
    if (!ofs.is_open()) {
        cerr << "open " << outPath << " error !" << endl;
        exit(-1);
    }


    VlCovDetFeature const * feature = (VlCovDetFeature const *) vl_covdet_get_features(covdet);
    VlSiftFilt * sift = vl_sift_new(16, 16, 1, 3, 0);
    vl_index i;
    vl_size dimension = 128;
    vl_size patchSide = 2 * patchResolution + 1;
    double patchStep = (double)patchRelativeExtent / patchResolution;

    vl_sift_set_magnif(sift, 3.0);
    numFeatures = vl_covdet_get_num_features(covdet);
    for (i = 0; i < (signed)numFeatures; ++i) {
        float desc[128];
        /* extract the patch */
        vl_covdet_extract_patch_for_frame(covdet,
                                          patch,
                                          patchResolution,
                                          patchRelativeExtent,
                                          patchRelativeSmoothing,
                                          feature[i].frame);

        /* compute gradient and angle on the patches */
        vl_imgradient_polar_f(patchXY, patchXY + 1,
                               2, 2 * patchSide,
                               patch, patchSide, patchSide, patchSide);


        /* compute the sift desc based on the gradient and angle */
        vl_sift_calc_raw_descriptor(sift,
                                     patchXY,
                                     desc,
                                     (int)patchSide, (int)patchSide,
                                     (double)(patchSide-1) / 2, (double)(patchSide-1) / 2,
                                     (double)patchRelativeExtent / (3.0 * (4 + 1) / 2) / patchStep,
                                     VL_PI / 2);

        float flipDesc[128];
        /* flip the feature descr */
        flip_descriptor(flipDesc, desc);
        /* print the feature vector */
        vl_index j = 0;
        ofs << fixed << setprecision(7) << flipDesc[j];
        for (j = 1; j < (signed)dimension; ++j) {
            ofs << " " << fixed << setprecision(7) << flipDesc[j];
        }
        ofs << endl;
    }

    if (sift) {
        vl_sift_delete(sift);
    }
    if (covdet) {
        vl_covdet_delete(covdet);
    }
    ofs.close();
    return 0;
}
