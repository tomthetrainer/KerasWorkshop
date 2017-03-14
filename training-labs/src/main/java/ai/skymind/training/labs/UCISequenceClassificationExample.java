package ai.skymind.training.labs;

import ai.skymind.training.solutions.UCIData;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.split.NumberedFileInputSplit;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Sequence Classification Example Using a LSTM Recurrent Neural Network
 * This example learns how to classify univariate time series as belonging to one of six categories.
 * Data is the UCI Synthetic Control Chart Time Series Data Set
 * Details:     https://archive.ics.uci.edu/ml/datasets/Synthetic+Control+Chart+Time+Series
 * Data:        https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/synthetic_control.data
 * Image:       https://archive.ics.uci.edu/ml/machine-learning-databases/synthetic_control-mld/data.jpeg
 *
 * @author Alex Black
 */
public class UCISequenceClassificationExample {

    private static final Logger log = LoggerFactory.getLogger(UCISequenceClassificationExample.class);

    public static void main(String[] args) throws Exception {


        /*
            ######## LAB STEP 1 ##############################################

            Download and write the data in a suitable format into csv files
                - separate files for features and labels
                - separate directories for train and test (a 75:25 split)
                Use for future reference: https://deeplearning4j.org/usingrnns#data

            ##################################################################
            */


        // ######### LAB STEP 1 #########
        //         ADD YOUR CODE HERE





        /*
            ######## LAB STEP 2 ##############################################

            STEP II. Set up training Data recordreaders
            Load the training data using csv record readers. Specify a minibatch size for the record reader.
            Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
            For future reference on csv record readers refer to the CSV record reader examples in the dl4j-examples repo
            ##################################################################
            */


        // ######### LAB STEP 2 #########
        //         ADD YOUR CODE HERE



        /*
            ######## LAB STEP 3 ##############################################

            STEP III. Set the miniBatchSize and the number of classes
            Specify a minibatch size for the record reader.

            ##################################################################
            */


        // ######### LAB STEP 3 #########
        //         ADD YOUR CODE HERE



        /*
            ######## LAB STEP 4 ##############################################

            Create a DataSetIterator to build the INDArrays that will be passed
            to the Neural Network

            ##################################################################
            */


        // ######### LAB STEP 4 #########
        //         ADD YOUR CODE HERE



        /*
            ######## LAB STEP 5 ##############################################

            Normalizing
            Here we use a standard normalizer that will subtract the mean and divide by the std dev
            ".fit" on data -> collects statistics (mean and std dev)
            ".setPreProcessor" -> allows us to use previously collected statistics to normalize on-the-fly.
            For future reference:
                Example in dl4j-examples with a min max normalizer

            ##################################################################
            */


        // ######### LAB STEP 5 #########
        //         ADD YOUR CODE HERE


            /*
            ######## LAB STEP 6 ##############################################


            STEP 6. Set up test data.
            Very important: apply the same normalization to the test and train.


            ##################################################################
            */


        // ######### LAB STEP 6 #########
        //         ADD YOUR CODE HERE




            /*
            ######## LAB STEP 7 ##############################################

            Configure the network and initialize it
            Note that the .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) is not always required,
                but is a technique that was found to help with this data set

            ##################################################################
            */


        // ######### LAB STEP 7 #########
        //         ADD YOUR CODE HERE



            /*
            ######## LAB STEP 8 ##############################################

            Add a UI

            ##################################################################
            */


        // ######### LAB STEP 8 #########
        //         ADD YOUR CODE HERE


            /*
            ######## LAB STEP 9 ##############################################


            Train the network, evaluating the test set performance at each epoch
                      Track the loss function and the weight changes and other metrics in the UI.
                      Open up: http://localhost:9000/

            ##################################################################
            */


        // ######### LAB STEP 9 #########
        //         ADD YOUR CODE HERE






    }

}
