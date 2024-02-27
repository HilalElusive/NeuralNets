package training;

import java.io.File;
import java.io.IOException;

import core.NeuralNetwork;
import data.*;
import functions.*;
import utils.*;

public class NetworkTrainer {
    private HyperParameters hyperParameters;

    private DataSetHelper.SplitDataSet splitDataSet;
    private Batch[] trainingBatches;
    private DataPoint[] allData;
    
    //Network State
	private NeuralNetwork neuralNetwork;
    private int epochCount;
	private int batchIndex;
	private double currentLearnRate;
	private double validationAccuracy;

	private boolean trainingActive = false;
	private double timeTaken;
	
    private String userHome = System.getProperty("user.home");

    public NetworkTrainer() {
        this.hyperParameters = new HyperParameters();
    }
    
    public void startTrainingSession() {
        loadMnistData();
        initializeNeuralNetwork();
        trainingActive = true;
        System.out.println("Training Session Ready to Start...");
    }
    
    public void run() {
        while (trainingActive && validationAccuracy < 95) {
            runTrainingCycle();
        }
    }
    
    public void save(String fileName) {
        String fullPath = new File(userHome + File.separator + "Downloads", fileName + ".json").getPath();
        try {
            NetworkSnapshot.saveToFile(neuralNetwork, fullPath);
            System.out.println("Model Saved Successfully.\nAt: " + fullPath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void runTrainingCycle() {
    	timeTaken = System.currentTimeMillis();
    	neuralNetwork.learn(trainingBatches[batchIndex].data, currentLearnRate, hyperParameters.regularization, hyperParameters.momentum);
    	batchIndex++;
    	
    	if (batchIndex >= trainingBatches.length) {
    		epochCompleted();
    	}
    }
    
    private void epochCompleted() {
        batchIndex = 0;
        epochCount++;
        DataSetHelper.shuffleBatches(trainingBatches);
        currentLearnRate = (1.0 / (1.0 + hyperParameters.learnRateDecay * epochCount)) * hyperParameters.initialLearningRate;
        
        // Evaluate the network's performance on the training dataset
        double trainingCost = calculateCost(splitDataSet.train);
        double trainingAccuracy = calculateAccuracy(splitDataSet.train);
        
        // Evaluate on the validation dataset if you want to monitor for overfitting
        double validationCost = calculateCost(splitDataSet.validate);
        validationAccuracy = calculateAccuracy(splitDataSet.validate);
        
        timeTaken = (System.currentTimeMillis() - timeTaken)/1000;

        System.out.println("Epoch " + epochCount + " completed. Time taken : " + timeTaken + "s.");
        System.out.println("Training Cost: " + String.format("%.4f", trainingCost) + ", Training Accuracy: " + String.format("%.3f", trainingAccuracy) + "%");
        System.out.println("Validation Cost: " + String.format("%.4f", validationCost) + ", Validation Accuracy: " + String.format("%.3f", validationAccuracy) + "%\n");
    }
    
    private double calculateCost(DataPoint[] dataSet) {
        double totalCost = 0.0;
        for (DataPoint dp : dataSet) {
            double[] outputs = neuralNetwork.CalculateOutputs(dp.inputs);
            totalCost += neuralNetwork.getCostFunction().costFunction(outputs, dp.expectedOutputs);
        }
        return totalCost / dataSet.length;
    }

    private double calculateAccuracy(DataPoint[] dataSet) {
        int correctPredictions = 0;
        for (DataPoint dp : dataSet) {
            double[] outputs = neuralNetwork.CalculateOutputs(dp.inputs);
            int predictedLabel = neuralNetwork.MaxValueIndex(outputs);
            if (predictedLabel == dp.label) {
                correctPredictions++;
            }
        }
        return 100.0 * correctPredictions / dataSet.length;
    }
    
    private void initializeNeuralNetwork() {
        neuralNetwork = new NeuralNetwork(hyperParameters.layerSizes);
        IActivation activation = Activation.getActivationFromType(hyperParameters.activationType);
        IActivation outputLayerActivation = Activation.getActivationFromType(hyperParameters.outputActivationType);
        neuralNetwork.setActivationFunction(activation, outputLayerActivation);
        neuralNetwork.setCostFunction(Cost.getCostFromType(hyperParameters.costType));
        currentLearnRate = hyperParameters.initialLearningRate;
    }

    private void loadMnistData() {
    	String filePath = new File(userHome, "Downloads" + File.separator + "mnist_dataset.csv").getPath();
        System.out.println("Reading The Dataset...");
        allData = CSVDataLoader.readCSV(filePath);
        splitDataSet = DataSetHelper.splitData(allData, hyperParameters.trainingSplit, true);
        trainingBatches = DataSetHelper.createMiniBatches(splitDataSet.train, hyperParameters.minibatchSize, true);
        System.out.println("Data Loaded.");
    }
    
    public NeuralNetwork getNeuralNetwork() {
		return neuralNetwork;
	}
    
    public DataPoint[] getValidationData() {
    	return splitDataSet.validate;
    }
}
