package ai.skymind.training.labs;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;

/*
 *

 *
 *
 *
 *
 */

public class ConvNeuralNet {

    // Provide a Logger
    private static Logger log = LoggerFactory.getLogger(ConvNeuralNet.class);


        public static void main(String[] args) throws Exception {
            // image information
            // 28 * 28 grayscale
            // grayscale implies single channel
            int height = 28;
            int width = 28;
            int channels = 1;

        // Neural Networks begin Training by setting Random weights
        // setting the random seed allows consistent results

            int rngseed = 123;
            Random randNumGen = new Random(rngseed);
            // batchSize determines how many records to train on before adjusting weights
            int batchSize = 128;

            // Outputnum = number of classes
            // in our case digits 0-9 or 10

            int outputNum = 10;

            // An epoch is a total pass through the dataset
            int numEpochs = 1;




        /*
        Define the File Paths
        use ClassPathResource("<STRING OF FILE PATH DIRECTORY").getFile();

        JAVADOC
        https://deeplearning4j.org/datavecdoc/org/datavec/api/util/ClassPathResource.html
        */

        File trainData = new ClassPathResource("mnist_png/training").getFile();
        File testData = new ClassPathResource("mnist_png/testing").getFile();


        /*
        Define the FileSplit(PATH, ALLOWED FORMATS,random)

        JAVADOC
        https://deeplearning4j.org/datavecdoc/org/datavec/api/split/FileSplit.html
         */

        FileSplit train = new FileSplit(trainData, NativeImageLoader.ALLOWED_FORMATS,randNumGen);
        FileSplit test = new FileSplit(testData,NativeImageLoader.ALLOWED_FORMATS,randNumGen);

        /*
         Extract the parent path as the image label

        JAVADOC
        https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/ParentPathLabelGenerator.html

         */


        ParentPathLabelGenerator labelMaker = new ParentPathLabelGenerator();

            /*
            Create a record reader, pass it the height,width,channels of the images,
            and the labelmaker that creates the label

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/ParentPathLabelGenerator.html
            */

        ImageRecordReader recordReader = new ImageRecordReader(height,width,channels,labelMaker);
        ImageRecordReader recordReaderTest = new ImageRecordReader(height,width,channels,labelMaker);

        /*
        Initialize the record reader
        optional test would be to add a listener, to extract the name
        and verify the labels

        JAVADOC
        https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html

        */

        recordReader.initialize(train);
        recordReaderTest.initialize(test);


        /*
         Record Reader returns a List of Writables, Writables are for serializing data
         A DataSetIterator takes the batch of writables and builds an INDArray
         to send to the Network

        Example DataFile.csv
        1.1,1.2,1.3
        2.1,2.2,2,3
        3.1,3.2,3,3

        RecordReader converts to List of Writable
        List<Float,Float,Float>

        DataSetIterator converts List to INDArray
        [[1.1,1.2,1.3
        2.1,2.2,2,3
        3.1,3.2,3,3]]

        JAVADOC
        http://nd4j.org/doc/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html

        */

        DataSetIterator dataIter = new RecordReaderDataSetIterator(recordReader,batchSize,1,outputNum);
        DataSetIterator testIter = new RecordReaderDataSetIterator(recordReaderTest,batchSize,1,outputNum);

        /*
        Pixel Values will be 0-255
        Neural Nets train better on values centered on 0, or with small range of 0-1
        In this case scale values between 0-1
        Using DataNormaliztion ImagePreProcessingScaler

        JAVADOC
        http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.html
         */

        //Define the scaler
        DataNormalization scaler = new ImagePreProcessingScaler(0,1);


        // call fit method of the scaler so it can read the data for min and max
        scaler.fit(dataIter);


        // apply the scaler using setPreProcessor method of the datasetiterator
        dataIter.setPreProcessor(scaler);
        testIter.setPreProcessor(scaler);

        // Build Our Neural Network

        log.info("**** Build Model ****");


        /*
        First Create a MultiLayerConfiguration object

        JAVADOC
        https://deeplearning4j.org/doc/org/deeplearning4j/nn/conf/MultiLayerConfiguration.html
         */


       /* <-- DELETE THIS LINE -->


        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .iterations(1)
            .learningRate(0.006)
            .updater(Updater.NESTEROVS).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            // ######### LAB STEP 2 #################
            //       REPLACE WITH THE CODE in STEP 2

            // ######### LAB STEP 3 #################
            //       REPLACE WITH THE CODE in STEP 3

            // ######### LAB STEP 4 #################
            //       REPLACE WITH THE CODE in STEP 4



                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation(Activation.SOFTMAX)
                        .build())
                .pretrain(false).backprop(true)
            .setInputType(InputType.convolutional(height,width,channels))
            .build();





        MultiLayerNetwork model = new MultiLayerNetwork(conf);

        // Initialize the model
        model.init();

        // attach a Listener
        model.setListeners(new ScoreIterationListener(10));


        // attach a UIServer



        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(1));
        uiServer.attach(statsStorage);


        // Train the model
        log.info("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataIter);
        }

        //testIter.reset();
        //DataSet t = testIter.next();
        //System.out.println(t);


            System.out.println("Evaluate model....");
            Evaluation eval = new Evaluation(outputNum);
            while(testIter.hasNext()){
                DataSet t = testIter.next();
                INDArray output = model.output(t.getFeatureMatrix()); //get the networks prediction
                eval.eval(t.getLabels(), output); //check the prediction against the true cla


            }
            log.info(eval.stats());
            log.info("****************Example finished********************");

       <-- DELETE THIS LINE --> */

    }


}
