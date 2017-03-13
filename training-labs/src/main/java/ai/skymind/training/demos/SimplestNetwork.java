package ai.skymind.training.demos;

import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ParamAndGradientIterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.InMemoryStatsStorage;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Random;

/**
 * Built for SkyMind Training class
 */
public class SimplestNetwork {
    private static Logger log = LoggerFactory.getLogger(SimplestNetwork.class);
    public static void main(String[] args) throws  Exception{
        /*
        Most Basic NN that takes a single input
         */

        int seed = 123; // consistent Random Numbers needed for testing, Initial weights are Randomized
        Random rng = new Random(seed);

        int nEpochs = 500; //Number of epochs (full passes of the data)

        double learningRate = 0.005; // How Fast to adjust weights to minimize error
        // Start with Learning Rate of 0.005

        int numInputs = 1; // number of input nodes

        int numOutputs = 1; // number of output nodes

        int nHidden = 5; // number of hidden nodes
        /*
        Create our input values and expected output values
        All data in all Neural Networks are represented as
        Numerical arrays, Normalization between 0 and 1 allows for better training
        */

        INDArray input = Nd4j.create(new float[]{(float) 0.5},new int[]{1,1}); // Our input value
        INDArray output = Nd4j.create(new float[]{(float) 0.8},new int[]{1,1}); // expected output
        log.info("******" + input.toString() + "*********" );

        /*
        Build a MuliLayer Network to train on our dataset
        */


        MultiLayerNetwork model = new MultiLayerNetwork(new NeuralNetConfiguration.Builder()
            .seed(seed)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT) // most commonly used Optimization algo
            .learningRate(learningRate)
            .weightInit(WeightInit.XAVIER) // Xavier is a weight randomizer optimized for NN
            .updater(Updater.NESTEROVS).momentum(0.09) // How to update the weights start with momentum of 0.09
            .list()
            .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(nHidden)
                .activation(Activation.TANH)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .activation(Activation.IDENTITY)
                .nIn(nHidden).nOut(numOutputs).build())
            .pretrain(false).backprop(true).build()
        );
        model.init();

        /*
        Create a web based UI server to show progress as the network trains
        The Listeners for the model are set here as well
        One listener to pass stats to the UI
        and a Listener to pass progress info to the console
         */

        UIServer uiServer = UIServer.getInstance();
        StatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(new StatsListener(statsStorage),new ScoreIterationListener(1));
        uiServer.attach(statsStorage);
        /*
        ParamAndGradientIterationListener pgl = ParamAndGradientIterationListener.builder()
                .printHeader(true)
                .delimiter("|")
                .outputToConsole(true)
                .printMean(true)
                .iterations(1)
                .build();

        model.setListeners(pgl);
        */


        //UIServer uiServer = UIServer.getInstance();

        //Configure where the network information (gradients, activations, score vs. time etc) is to be stored
        //Then add the StatsListener to collect this information from the network, as it trains
        //StatsStorage statsStorage = new InMemoryStatsStorage();             //Alternative: new FileStatsStorage(File) - see UIStorageExample
        //int listenerFrequency = 1;
        //model.setListeners(new StatsListener(statsStorage, listenerFrequency));

        //Attach the StatsStorage instance to the UI: this allows the contents of the StatsStorage to be visualized
        //uiServer.attach(statsStorage);


        //Train the network on the full data set, and evaluate in periodically
        for( int i=0; i<nEpochs; i++ ){
            model.fit(input,output);
            INDArray params = model.params();
            System.out.println(params);
            INDArray output2 = model.output(input);
            log.info(output2.toString());
            Thread.sleep(100);
        }

    }
}
