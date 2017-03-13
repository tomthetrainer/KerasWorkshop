package ai.skymind.training.demos;

/**
 * Created by tomhanlon on 2/21/17.
 */
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;

/*
  * SIMPLE DEMO to load a 4*4 black and white image into an array
  * Then normalize the array
 */
public class ImageDemo {
    public static void main(String[]args) throws Exception{
        NativeImageLoader loader = new NativeImageLoader(4, 4, 1);
        File image = new ClassPathResource("44.png").getFile();
        INDArray imagematrix = loader.asMatrix(image);

        System.out.println("Raw image as matrix");
        System.out.println(imagematrix);

        DataNormalization scaler = new ImagePreProcessingScaler(0,1);
        scaler.transform(imagematrix);

        System.out.println("Scaled image");
        System.out.println(imagematrix);

    }
}