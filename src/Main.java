import java.io.File;
import java.io.IOException;

import core.NeuralNetwork;
import data.CSVDataLoader;
import data.DataPoint;
import utils.NetworkSnapshot;

public class Main {
	
    public static void main(String[] args) throws IOException {	
    	
    	//Loading our Trained Model
        String userHome = System.getProperty("user.home");
        String fullPath = new File(userHome + File.separator + "Downloads", "model.json").getPath();
        NeuralNetwork nn = NetworkSnapshot.loadNetworkFromFile(fullPath);
        
        //Loading the Validation Dataset To Check the Model's Accuracy
        String filePath = new File(userHome, "Downloads" + File.separator + "mnist_test.csv").getPath();
        System.out.println("Reading The Validation Dataset...");
        DataPoint[] a = CSVDataLoader.readCSV(filePath);
        
        for (int i = 0; i < 40; i++) {
        	System.out.println("Actual Label: " + a[i].label + " " +
					"Predicted label: " + nn.MaxValueIndex(nn.CalculateOutputs(a[i].inputs)));
        }
    }
}

		/*NeuralNetwork nn = networkTrainer.getNeuralNetwork();

		// Instantiate and start the network trainer
        NetworkTrainer networkTrainer = new NetworkTrainer();
        networkTrainer.startTrainingSession();
        networkTrainer.run();

        System.out.println("Training Session Completed.");
        
        networkTrainer.save("model"); //Model Saved Automatically in the Download folder*/
