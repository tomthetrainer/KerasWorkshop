package ai.skymind.training.solutions;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;

/**
 * Created by tomhanlon on 1/13/17.
 */
public class ImportVGG16 {
    private static Logger log = LoggerFactory.getLogger(ImportVGG16.class);
    public static void main(String[] args) throws Exception{


        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/vgg_save_config","/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/vgg_save.h5");
        ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/tmp/vgg_combined_save.h5",true);


        // img = image.load_img(img_path, target_size=(224, 224))

        int height = 224;
        int width = 224;
        int channels = 3;
        File file = new ClassPathResource("elephant.jpeg").getFile();
        //File file = new File("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/elephant.jpg");

        NativeImageLoader loader = new NativeImageLoader(height, width, channels);

        // Get the image into an INDarray
        // Transform it as the original were transformed
        // Subtract mean from pixel values

        INDArray image = loader.asMatrix(file);
        DataNormalization scaler = new VGG16ImagePreProcessor();
        scaler.transform(image);

        // Get the output
        INDArray[] output = model.output(image);


        // Output is probability for 1000 labels
        System.out.println(output[0].data());
        System.out.println(output.length);
        System.out.println(output[0].rank());
        //System.out.println(output[0].maxComplex().toString());
        double maxValue = output[0].maxNumber().doubleValue();
        System.out.println(maxValue);
        //INDArray maxAlong0 = output[0].max(0);
        System.out.println(output[0].rows());
        System.out.println(output[0].columns());
        //System.out.println(maxAlong0);
        // System.out.println(output[0].getRow(0)); // all 0's
        // double sum = Nd4j.getExecutioner().execAndReturn(new Sum(output[0])).getFinalResult().doubleValue();
        //   System.out.println(sum);
        INDArray answers = Nd4j.argMax(output[0],1);
        System.out.println(answers);
         /*
            Output
            386.00
            branch A
            Final index: 386;
             */

        //System.out.println(Nd4j.argMax(output[0],Integer.MAX_VALUE));
        //INDArray values = output[0];
        //System.out.println(values);
        //sortWithIndices(INDArray ndarray, int dimension, boolean ascending)

        INDArray[] sorted = Nd4j.sortWithIndices(output[0],1,false);
        System.out.println(sorted[0].data());
        //System.out.println(TrainedModels.VGG16.decodePredictions(output[0](0)));

        INDArray[] output2 = model.output(image);
        String predictions = TrainedModels.VGG16.decodePredictions(output2[0]);
        System.out.println(predictions);

        //System.out.println(image);
    }
}
