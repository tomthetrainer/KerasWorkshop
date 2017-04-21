package ai.skymind.training.solutions;

import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;


/**
 * Created by tomhanlon on 4/20/17.
 */
public class KerasModelimportInceptionV3 {


    public static void main(String[] args) throws Exception{

        int imgWidth = 299;
        int imgHeight = 299;
        int imgChannels = 3;
        int numClasses = 1000;


        //Path to Saved Model and weights
        // We use DataVec's ClassPathResource here, you could pass the import files path string directly

        //String kerasModelfromKerasExport = new ClassPathResource("inception_v3_complete.h5").getFile().getPath();

        /*
        Create a MultiLayerNetwork from the saved model
         */

        // MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);

        // ComputationGraph model = KerasModelImport.importKerasModelAndWeights(kerasModelfromKerasExport);
        //ComputationGraph model = KerasModelImport.importKerasModelAndWeights(kerasModelfromKerasExport);
        //ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/inception_V3_config","/Users/tomhanlon/tensorflow/vgg16/keras-model-zoo/deep-learning-models/inception_v3.h5",false);

        ComputationGraph model = KerasModelImport.importKerasModelAndWeights("/tmp/inception_v3_complete.h5");



        File elephant = new ClassPathResource("elephant.jpeg").getFile();
       //NativeImageLoader loader = new NativeImageLoader();

        NativeImageLoader imageLoader = new NativeImageLoader(imgHeight, imgWidth, imgChannels);
        INDArray image = imageLoader.asMatrix(elephant).div(255.0).sub(0.5).mul(2);
        //Function<INDArray, INDArray> preProcessor = image2 -> image.div(255.0).sub(0.5).mul(2);




        //preProcessorreProcessor.apply(imageLoader.asMatrix(imageStream));
       //preProcessor.apply(image);

        // INDArray imgData = preProcess(imageFile.getPath());

        //INDArray imgData = net.preProcessImage(imageFile.getPath());



        INDArray[] output = model.output(false,image);

        //INDArray image = loader.asMatrix(elephant);
        //System.out.print(image);
        //DataNormalization scaler = new
        //scaler.transform(image);
        //DataNormalization scaler = new VGG16ImagePreProcessor();
        //scaler.transform(image);
        //System.out.print(image);
        //INDArray[] output = model.output(false,image);
        System.out.println(TrainedModels.VGG16.decodePredictions(output[0]));

        //System.out.println(output);

    }

}

