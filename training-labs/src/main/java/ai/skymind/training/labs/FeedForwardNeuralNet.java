package ai.skymind.training.labs;

import org.datavec.api.io.labels.ParentPathLabelGenerator;
import org.datavec.api.split.FileSplit;
import org.datavec.api.util.ClassPathResource;
import org.datavec.image.loader.NativeImageLoader;
import org.datavec.image.recordreader.ImageRecordReader;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Random;




public class FeedForwardNeuralNet {

    // Provide a Logger
    private static Logger log = LoggerFactory.getLogger(FeedForwardNeuralNet.class);


        public static void main(String[] args) throws Exception {

            /*
            ######## LAB STEP 2 ##############################################

            Add 3 lines to define 3 Integers
            the height, width and depth of the images
            28,28, and 1
            Note that images will be scaled if they do not fit those measurements
            ##################################################################
            */


            // ######### LAB STEP 2 #########
            //         ADD YOUR CODE HERE



            /*
            ######## LAB STEP 3 ##############################################

            Set an integer seed and Random  by calling new Random(Your_Seed)
            ##################################################################
             */


            // ######### LAB STEP 3 ##########
            //         ADD YOUR CODE HERE




            /*
            ######## LAB STEP 4 ##############################################

            Set an int batchSize to the value of 128
            ##################################################################
            */


            // ######### LAB STEP 4 ##########
            //         ADD YOUR CODE HERE






            /*
            ######## LAB STEP 5 ##############################################

            Set an int outputNum to the value of 10
            ##################################################################
             */


            // ######### LAB STEP 5 ##########
            //         ADD YOUR CODE HERE



            /*
            ######## LAB STEP 6 ##############################################

            Set numEpochs to 15
            ##################################################################
            */

            // ######### LAB STEP 6 ##########
            //         ADD YOUR CODE HERE








            /*
            ######## LAB STEP 7 ##############################################


            Define file paths for train directory and test directory
            use ClassPathResource("<STRING OF FILE PATH DIRECTORY").getFile();

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/util/ClassPathResource.html
            ##################################################################
            */


            // ######### LAB STEP 7 ##########
            //         ADD YOUR CODE HERE






            /*
            ######## LAB STEP 8 ##############################################

            Define FileSplit for File Path
            FileSplit(PATH, ALLOWED FORMATS,random)

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/split/FileSplit.html
            ##################################################################
            */


            // ######### LAB STEP 8 ##########
            //         ADD YOUR CODE HERE






            /*
            ######## LAB STEP 9 ##############################################

            Create a Parent Path Label Generator to label the images 0-9

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/io/labels/ParentPathLabelGenerator.html
            ##################################################################
            */


            // ######### LAB STEP 9 ##########
            //         ADD YOUR CODE HERE







            /*
            ######## LAB STEP 10 ##############################################


            Create two ImageRecordReaders one for test one for train
            pass them height, width, depth, and labelmaker

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html
            ##################################################################
            */

            // ######### LAB STEP 10 ##########
            //         ADD YOUR CODE HERE





            /*
            ######## LAB STEP 11 ##############################################

            Initialize the Record Readers

            JAVADOC
            https://deeplearning4j.org/datavecdoc/org/datavec/api/records/reader/impl/csv/CSVRecordReader.html
            ##################################################################
            */

            // ######### LAB STEP 11 ##########
            //         ADD YOUR CODE HERE






            /*
            ######## LAB STEP 12 ##############################################

            Create DataSetIterator to take list of Writables and build INDArray

            JAVADOC
            http://nd4j.org/doc/org/nd4j/linalg/dataset/api/iterator/DataSetIterator.html
            ##################################################################
            */

            // ######### LAB STEP 12 ##########
            //         ADD YOUR CODE HERE







            /*
            ######## LAB STEP 13 ##############################################

            Scale the Pixel Values to 0-1
            Pixel Values will originally  be 0-255
            Neural Nets train better on values centered on 0, or with small range of 0-1
            In this case scale values between 0-1
            Use a DataNormaliztion ImagePreProcessingScaler

            JAVADOC
            http://nd4j.org/doc/org/nd4j/linalg/dataset/api/preprocessor/ImagePreProcessingScaler.html
            ##################################################################
            */


            // ######### LAB STEP 13 ##########
            //         ADD YOUR CODE HERE



            /*
            ######## LAB STEP 14 ##############################################

            call fit method of the scaler so it can read the data for min and max
            scaler.fit(dataIter);
            ##################################################################
            */

            // ######### LAB STEP 14 ##########
            //         ADD YOUR CODE HERE


            /*
            ######## LAB STEP 15 ##############################################

            apply the scaler using setPreProcessor method of the datasetiterator
            dataIter.setPreProcessor(scaler);
            ##################################################################
            */




            // ######### LAB STEP 15 ##########
            //         ADD YOUR CODE HERE


            // Build Our Neural Network

            log.info("**** Build Model ****");




           /*
            ######## LAB STEPS 16,17,18,19 ####################################

            To complete these steps
            Uncomment the code block below, from MultilayerCiguration, to build()


            ##################################################################
            */




       /*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(rngseed)
            .optimizationAlgo(OptimizationAlgorithm.### YOUR CODE HERE ####)
            .iterations(1)
            .learningRate(### YOUR CODE HERE ####)
            .updater(Updater.### YOUR CODE HERE ####).momentum(0.9)
            .regularization(true).l2(1e-4)
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(### YOUR CODE HERE ####)
                .nOut(100)
                .activation(Activation.### YOUR CODE HERE ####)
                .weightInit(WeightInit.### YOUR CODE HERE ####)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                .nIn(### YOUR CODE HERE ####)
                .nOut(### YOUR CODE HERE ####)
                .activation(Activation.### YOUR CODE HERE ####)
                .weightInit(WeightInit.### YOUR CODE HERE ####)
                .build())
            .pretrain(### YOUR CODE HERE ####).backprop(### YOUR CODE HERE ####)
            .setInputType(InputType.convolutional(height,width,channels))
            .build();


        */


            /*
            ######## LAB STEP 20 ##############################################
            Uncomment the code below by Deleting
            the 2 lines that contain
            <-- remove this line -->

            ##################################################################*/

            /* <-- remove this line -->

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        model.setListeners(new ScoreIterationListener(10)); // attach a listener


         //attach a UIServer

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(1));
        uiServer.attach(statsStorage);


        // Train the model
        log.info("*****TRAIN MODEL********");
        for(int i = 0; i<numEpochs; i++){
            model.fit(dataIter);
        }

         <-- remove this line --> */

    }


 }
