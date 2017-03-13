package ai.skymind.training.solutions;

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
            STEP I.
            Download and write the data in a suitable format into csv files
                - separate files for features and labels
                - separate directories for train and test (a 75:25 split)
                Use for future reference: https://deeplearning4j.org/usingrnns#data
         */

        UCIData.download();

        /*
            STEP II. Set up training Data
            Load the training data using csv record readers. Specify a minibatch size for the record reader.
            Note that we have 450 training files for features: train/features/0.csv through train/features/449.csv
            For future reference on csv record readers refer to the CSV record reader examples in the dl4j-examples repo
         */

        SequenceRecordReader trainFeatures = new CSVSequenceRecordReader();
        trainFeatures.initialize(new NumberedFileInputSplit(UCIData.featuresDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));
        SequenceRecordReader trainLabels = new CSVSequenceRecordReader();
        trainLabels.initialize(new NumberedFileInputSplit(UCIData.labelsDirTrain.getAbsolutePath() + "/%d.csv", 0, 449));

        int miniBatchSize = 10;
        int numLabelClasses = 6;
        DataSetIterator trainData = new SequenceRecordReaderDataSetIterator(trainFeatures, trainLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);

        /*
            STEP III. Normalizing
            Here we use a standard normalizer that will subtract the mean and divide by the std dev
            ".fit" on data -> collects statistics (mean and std dev)
            ".setPreProcessor" -> allows us to use previously collected statistics to normalize on-the-fly.
            For future reference:
                Example in dl4j-examples with a min max normalizer
         */
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(trainData);
        trainData.setPreProcessor(normalizer);

        /*
            STEP IV. Set up test data.
            Very important: apply the same normalization to the test and train.
         */
        SequenceRecordReader testFeatures = new CSVSequenceRecordReader();
        testFeatures.initialize(new NumberedFileInputSplit(UCIData.featuresDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        SequenceRecordReader testLabels = new CSVSequenceRecordReader();
        testLabels.initialize(new NumberedFileInputSplit(UCIData.labelsDirTest.getAbsolutePath() + "/%d.csv", 0, 149));
        DataSetIterator testData = new SequenceRecordReaderDataSetIterator(testFeatures, testLabels, miniBatchSize, numLabelClasses,
                false, SequenceRecordReaderDataSetIterator.AlignmentMode.ALIGN_END);
        testData.setPreProcessor(normalizer);

        /*
            STEP V.
            Configure the network and initialize it
            Note that the .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue) is not always required,
                but is a technique that was found to help with this data set
         */
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .weightInit(WeightInit.XAVIER)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .learningRate(0.005)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(0.5)
                .list()
                .layer(0, new GravesLSTM.Builder().activation("tanh").nIn(1).nOut(10).build())
                .layer(1, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MCXENT)
                        .activation("softmax").nIn(10).nOut(numLabelClasses).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        /*
            STEP VI. Set up the UI

         */
        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        net.setListeners(new StatsListener(statsStorage));
        uiServer.attach(statsStorage);


        /*
            STEP VII. Train the network, evaluating the test set performance at each epoch
                      Track the loss function and the weight changes and other metrics in the UI.
                      Open up: http://localhost:9000/
         */
        int nEpochs = 40;
        String str = "Test set evaluation at epoch %d: Accuracy = %.2f, F1 = %.2f";
        for (int i = 0; i < nEpochs; i++) {
            net.fit(trainData);
            Evaluation evaluation = net.evaluate(testData);
            log.info(String.format(str, i, evaluation.accuracy(), evaluation.f1()));
            testData.reset();
            trainData.reset();
        }
        log.info("----- Example Complete -----");
    }

}
