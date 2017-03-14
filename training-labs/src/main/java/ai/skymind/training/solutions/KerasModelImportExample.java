package ai.skymind.training.solutions;

import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;


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


        //Path to Saved Model and weights
        // We use DataVec's ClassPathResource here, you could pass the import files path string directly

        String kerasModelfromKerasExport = new ClassPathResource("Keras/full_iris_model").getFile().getPath();

        /*
        Create a MultiLayerNetwork from the saved model
         */

       // MultiLayerNetwork model = org.deeplearning4j.nn.modelimport.keras.KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);

        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(kerasModelfromKerasExport);
        /*
        The Model trained on Iris data, 4 fields
        Sepal Length, Sepal Width, Petal Length, Petal Width
        When asked to predict the class for the following input

        prediction = model.predict(numpy.array([[4.6,3.6,1.0,0.2]]));

        Output was...
        [[ 0.92084521  0.13397516  0.03294737]]

        To verify the output is proper for the loaded model test with the same data
        Input [4.60, 3.60, 1.00, 0.20]
        Output[0.92, 0.13, 0.03]
         */

        INDArray myArray = Nd4j.zeros(1, 4); // one row 4 column array
        myArray.putScalar(0,0, 4.6);
        myArray.putScalar(0,1, 3.6);
        myArray.putScalar(0,2, 1.0);
        myArray.putScalar(0,3, 0.2);

        INDArray output = model.output(myArray);
        System.out.println("First Model Output");
        System.out.println(myArray);
        System.out.println(output);

    }
}
