package ai.skymind.training.demos;

import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.modelimport.keras.trainedmodels.TrainedModels;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.VGG16ImagePreProcessor;

import javax.servlet.MultipartConfigElement;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;

import static spark.Spark.*;

//import org.nd4j.linalg.dataset.api.preprocessor.

/**
 * Created by tomhanlon on 1/25/17.
 */

public class VGG16SparkJavaWebApp {
    public static void main(String[] args) throws Exception {

        /*
        Demonstration instructions
        This takes at least 4 minutes to load
        When loaded You will see jetty activity in the log
        Point browser at http://localhost:4567/VGGpredict
        And load an image into the form
         */

        // Load Neural Network from serialized format
        //File savedNetwork = new ClassPathResource("vgg16.zip").getFile();

        File savedNetwork = new File("/tmp/vgg16.zip");
        ComputationGraph vgg16 = ModelSerializer.restoreComputationGraph(savedNetwork);


        // make upload directory to store loaded images
        File uploadDir = new File("upload");
        uploadDir.mkdir(); // create the upload directory if it doesn't exist


        // form to allow user to choose image to upload
        String form = "<form method='post' action='getPredictions' enctype='multipart/form-data'>\n" +
                "    <input type='file' name='uploaded_file'>\n" +
                "    <button>Upload picture</button>\n" +
                "</form>";

        // spark java settings to display form or results
        staticFiles.location("/Users/tomhanlon/SkyMind/webcontent"); // Static files
        get("/hello", (req, res) -> "Hello World");
        get("VGGpredict", (req, res) -> form);
        //post("getPredictions",(req, res) -> "GET RESULTS");

        post("/getPredictions", (req, res) -> {
            Path tempFile = Files.createTempFile(uploadDir.toPath(), "", "");
            req.attribute("org.eclipse.jetty.multipartConfig", new MultipartConfigElement("/temp"));

            try (InputStream input = req.raw().getPart("uploaded_file").getInputStream()) { // getPart needs to use same "name" as input field in form
                Files.copy(input, tempFile, StandardCopyOption.REPLACE_EXISTING);
            }


            File file = tempFile.toFile();

            // define native image loaders
            NativeImageLoader loader = new NativeImageLoader(224, 224, 3);
            INDArray image = loader.asMatrix(file);

            // Scale image in same manner as network was trained on
            DataNormalization scaler = new VGG16ImagePreProcessor();
            scaler.transform(image);
            file.delete();
            INDArray[] output = vgg16.output(false,image);
            // just added
            //Map<String, INDArray> mine = vgg16.feedForward();
            //System.out.println(mine);
            // just added
            String predictions = TrainedModels.VGG16.decodePredictions(output[0]);

            return "<h1> '" + predictions  + "' </h1>" +
                    "Would you like to try another" +
                    form;

            //return "<h1>Your image is: '" + tempFile.getName(1).toString() + "' </h1>";


        });

    }

}
