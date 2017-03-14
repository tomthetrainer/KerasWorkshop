package ai.skymind.training.labs;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


/*
            Keras Model Import Lab
            This Lab demonstrates importing
            a model into DeepLearning4J that had been
            saved in Keras

            WHERE IS THE PYTHON?
            If you look at the resources folder you will find
            Keras/iris.py

            WHAT DATA IS IT READING?
            Take a look at Keras/iris.csv

            WHY IS THIS A TOY EXAMPLE?
            In the interest of a small download and quick running it has been kept small.

            WHAT ABOUT THE BIG NETWORKS?
            Deeplearning4J can import large convolutional Neural Networks from Keras

            See our video on the topic
            https://www.youtube.com/watch?v=Cran8wsZLN4

            See a demo of VGG-16 doing image recognition
            https://deeplearning4j.org/demo-classifier-vgg16

            See the documentation of how we created that demo
            https://deeplearning4j.org/build_vgg_webapp

            JAVADOC for Keras Model Import
            https://deeplearning4j.org/doc/org/deeplearning4j/nn/modelimport/keras/KerasModelImport.html

         */

public class KerasModelImportExample {
    public static void main(String[] args) throws Exception{

         /*
            ######## LAB STEP 1 ##############################################

            Using ClassPathResource create a String for the path to
            the file where Keras has saved the model
            ##################################################################
            */


        // ######### LAB STEP 1 #########
        //         ADD YOUR CODE HERE



         /*
            ######## LAB STEP 2 ##############################################

            Create a DeepLearning4J MultiLayerNetwork by using
            org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights
            ##################################################################
            */


        // ######### LAB STEP 2 #########
        //         ADD YOUR CODE HERE



        /*
          ######## LAB STEP 3 ##############################################

          Verify consistent results over both the Keras and DeepLearning4J
          Version of the model by passing in a test array for inference
          In this step you build the INDArray
          ##################################################################
           */


        // ######### LAB STEP 3 #########
        //         ADD YOUR CODE HERE




        /*
          ######## LAB STEP 4 ##############################################

          Pass the validation case built in Step 3 to the model and print the
          output
          ##################################################################
           */


        // ######### LAB STEP 4 #########
        //         ADD YOUR CODE HERE



    }
}
