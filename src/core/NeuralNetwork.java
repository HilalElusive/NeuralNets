package core;

import java.util.Random;
import java.util.stream.IntStream;

import data.DataPoint;
import functions.*;
import training.LayerLearnData;
import training.NetworkLearnData;

public class NeuralNetwork {
	
	private Layer[] layers;
	private int[] layerSizes;

	private ICost cost;
	private Random rng;
	NetworkLearnData[] batchLearnData;

	// Create the neural network
	public NeuralNetwork(int[] layerSizes) {
		this.setLayerSizes(layerSizes);
		rng = new Random();
		layers = new Layer[layerSizes.length - 1];
		
		for(int i = 0; i < layers.length; i++) {
			layers[i] = new Layer(layerSizes[i], layerSizes[i + 1], rng);
		}
		cost = new Cost.CrossEntropy();
	}

	public NeuralNetwork() {}

	// Run the inputs through the network to calculate the outputs
	public double[] CalculateOutputs(double[] inputs) {
		for(Layer layer : layers) {
			inputs = layer.calculateOutputs(inputs);
		}
		return inputs;
	}
	
	public void learn(DataPoint[] trainingData, double learnRate, double regularization, double momentum) {
	    if (batchLearnData == null || batchLearnData.length != trainingData.length) {
	        batchLearnData = new NetworkLearnData[trainingData.length];
	        for (int i = 0; i < batchLearnData.length; i++) {
	            batchLearnData[i] = new NetworkLearnData(layers);
	        }
	    }

	    // Using Java's parallel stream for concurrency
	    IntStream.range(0, trainingData.length).parallel().forEach(i -> {
	        updateGradients(trainingData[i], batchLearnData[i]);
	    });

	    // Update weights and biases based on the calculated gradients
	    for (int i = 0; i < layers.length; i++) {
	        layers[i].applyGradients(learnRate / trainingData.length, regularization, momentum);
	    }
	}

	void updateGradients(DataPoint data, NetworkLearnData learnData) {
	    // Feed data through the network to calculate outputs.
	    double[] inputsToNextLayer = data.inputs;

	    for (int i = 0; i < layers.length; i++) {
	        inputsToNextLayer = layers[i].calculateOutputs(inputsToNextLayer, learnData.layerData[i]);
	    }

	    // -- Backpropagation --
	    int outputLayerIndex = layers.length - 1;
	    Layer outputLayer = layers[outputLayerIndex];
	    LayerLearnData outputLearnData = learnData.layerData[outputLayerIndex];

	    // Update output layer gradients
	    outputLayer.calculateOutputLayerNodeValues(outputLearnData, data.expectedOutputs, cost);
	    outputLayer.updateGradients(outputLearnData);

	    // Update all hidden layer gradients
	    for (int i = outputLayerIndex - 1; i >= 0; i--) {
	        LayerLearnData layerLearnData = learnData.layerData[i];
	        Layer hiddenLayer = layers[i];

	        hiddenLayer.calculateHiddenLayerNodeValues(layerLearnData, layers[i + 1], learnData.layerData[i + 1].nodeValues);
	        hiddenLayer.updateGradients(layerLearnData);
	    }
	}

	public void setActivationFunction(IActivation activation) {
		setActivationFunction(activation, activation);
	}

	public void setActivationFunction(IActivation activation, IActivation outputLayerActivation) {
		for(int i = 0; i < layers.length - 1; i++) {
			layers[i].setActivationFunction(activation);
		}
		layers[layers.length - 1].setActivationFunction(outputLayerActivation);
	}


	public int MaxValueIndex(double[] values) {
		double maxValue = Double.MIN_VALUE;
		int index = 0;
		for (int i = 0; i < values.length; i++) {
			if (values[i] > maxValue) {
				maxValue = values[i];
				index = i;
			}
		}

		return index;
	}
	
	public void setCostFunction(ICost costFunction) {
		this.cost = costFunction;
	}
	
	public ICost getCostFunction() {
		return cost;
	}

	public int[] getLayerSizes() {
		return layerSizes;
	}
	
	public Layer[] getLayers() {
		return layers;
	}

	public void setLayerSizes(int[] layerSizes) {
		this.layerSizes = layerSizes;
	}
}
