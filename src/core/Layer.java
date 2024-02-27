package core;

import java.util.Random;
import functions.*;
import training.LayerLearnData;

public class Layer {
    private final int numNodesIn;
    private final int numNodesOut;

    private final double[] weights;
    private final double[] biases;

    // Cost gradient with respect to weights and with respect to biases
    private final double[] costGradientW;
    private final double[] costGradientB;

    // Used for adding momentum to gradient descent
    private final double[] weightVelocities;
    private final double[] biasVelocities;

    private IActivation activation;

    // Create the layer
    public Layer(int numNodesIn, int numNodesOut, Random rng) {
        this.numNodesIn = numNodesIn;
        this.numNodesOut = numNodesOut;
        this.activation = new Activation.Sigmoid();

        weights = new double[numNodesIn * numNodesOut];
        costGradientW = new double[weights.length];
        biases = new double[numNodesOut];
        costGradientB = new double[biases.length];

        weightVelocities = new double[weights.length];
        biasVelocities = new double[biases.length];

        initializeRandomWeights(rng);
    }

    // Calculate layer output activations
    public double[] calculateOutputs(double[] inputs) {
        double[] weightedInputs = new double[numNodesOut];

        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
            double weightedInput = biases[nodeOut];

            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
                weightedInput += inputs[nodeIn] * getWeight(nodeIn, nodeOut);
            }
            weightedInputs[nodeOut] = weightedInput;
        }

        // Apply activation function
        double[] activations = new double[numNodesOut];
        for (int outputNode = 0; outputNode < numNodesOut; outputNode++) {
            activations[outputNode] = activation.activate(weightedInputs, outputNode);
        }

        return activations;
    }
    
    // Calculate layer output activations and store inputs/weightedInputs/activations in the given learnData object
 	public double[] calculateOutputs(double[] inputs, LayerLearnData learnData) {
 		learnData.inputs = inputs;

 		for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
 			double weightedInput = biases[nodeOut];
 			for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
 				weightedInput += inputs[nodeIn] * getWeight(nodeIn, nodeOut);
 			}
 			learnData.weightedInputs[nodeOut] = weightedInput;
 		}

 		// Apply activation function
 		for (int i = 0; i < learnData.activations.length; i++) {
 			learnData.activations[i] = activation.activate(learnData.weightedInputs, i);
 		}

 		return learnData.activations;
 	}
 	
    // Update weights and biases based on previously calculated gradients.
    // Also resets the gradients to zero.
    public void applyGradients(double learnRate, double regularization, double momentum) {
        double weightDecay = (1 - regularization * learnRate);

        for (int i = 0; i < weights.length; i++) {
            double weight = weights[i];
            double velocity = weightVelocities[i] * momentum - costGradientW[i] * learnRate;
            weightVelocities[i] = velocity;
            weights[i] = weight * weightDecay + velocity;
            costGradientW[i] = 0;
        }

        for (int i = 0; i < biases.length; i++) {
            double velocity = biasVelocities[i] * momentum - costGradientB[i] * learnRate;
            biasVelocities[i] = velocity;
            biases[i] += velocity;
            costGradientB[i] = 0;
        }
    }
    
    public void calculateOutputLayerNodeValues(LayerLearnData layerLearnData, double[] expectedOutputs, ICost cost) {
		for (int i = 0; i < layerLearnData.nodeValues.length; i++) {
			double costDerivative = cost.costDerivative(layerLearnData.activations[i], expectedOutputs[i]);
			double activationDerivative = activation.derivative(layerLearnData.weightedInputs, i);
			layerLearnData.nodeValues[i] = costDerivative * activationDerivative;
		}
	}

	public void calculateHiddenLayerNodeValues(LayerLearnData layerLearnData, Layer oldLayer, double[] oldNodeValues) {
		for (int newNodeIndex = 0; newNodeIndex < numNodesOut; newNodeIndex++) {
			double newNodeValue = 0;
			
			for (int oldNodeIndex = 0; oldNodeIndex < oldNodeValues.length; oldNodeIndex++) {
				double weightedInputDerivative = oldLayer.getWeight(newNodeIndex, oldNodeIndex);
				newNodeValue += weightedInputDerivative * oldNodeValues[oldNodeIndex];
			}
			newNodeValue *= activation.derivative(layerLearnData.weightedInputs, newNodeIndex);
			layerLearnData.nodeValues[newNodeIndex] = newNodeValue;
		}

	}
	
	public void updateGradients(LayerLearnData layerLearnData) {
	    // Update cost gradient with respect to weights (synchronize for multithreading)
	    synchronized (costGradientW) {
	        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
	            double nodeValue = layerLearnData.nodeValues[nodeOut];
	            for (int nodeIn = 0; nodeIn < numNodesIn; nodeIn++) {
	                // Evaluate the partial derivative: cost / weight of current connection
	                double derivativeCostWrtWeight = layerLearnData.inputs[nodeIn] * nodeValue;
	                costGradientW[getFlatWeightIndex(nodeIn, nodeOut)] += derivativeCostWrtWeight;
	            }
	        }
	    }

	    synchronized (costGradientB) {
	        for (int nodeOut = 0; nodeOut < numNodesOut; nodeOut++) {
	            // Evaluate partial derivative: cost / bias
	            double derivativeCostWrtBias = 1 * layerLearnData.nodeValues[nodeOut];
	            costGradientB[nodeOut] += derivativeCostWrtBias;
	        }
	    }
	}

	public double getWeight(int nodeIn, int nodeOut) {
        int flatIndex = nodeOut * numNodesIn + nodeIn;
        return weights[flatIndex];
    }
	
	public int getFlatWeightIndex(int inputNeuronIndex, int outputNeuronIndex) {
		return outputNeuronIndex * numNodesIn + inputNeuronIndex;
	}

    public void setActivationFunction(IActivation activation) {
        this.activation = activation;
    }

    private void initializeRandomWeights(Random rng) {
        for (int i = 0; i < weights.length; i++) {
            weights[i] = randomInNormalDistribution(rng, 0, 1);
        }
    }

    private double randomInNormalDistribution(Random rng, double mean, double standardDeviation) {
        double x1 = 1 - rng.nextDouble();
        double x2 = 1 - rng.nextDouble();

        double y1 = Math.sqrt(-2.0 * Math.log(x1)) * Math.cos(2.0 * Math.PI * x2);
        return y1 * standardDeviation + mean;
    }
  
	public int getNumNodesOut() {
		return numNodesOut;
	}
	
	public IActivation getActivation() {
		return activation;
	}
	
	public double[] getWeights() {
		return weights;
	}

	public double[] getBiases() {
		return biases;
	}
}
