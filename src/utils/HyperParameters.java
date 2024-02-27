package utils;

import functions.*;

public class HyperParameters {
	//Network achitecture
	public int[] layerSizes;
	public Activation.ActivationType activationType;
	public Activation.ActivationType outputActivationType;
	public Cost.CostType costType;

	//Learning parameters
	public float trainingSplit;
	public double initialLearningRate;
	public double learnRateDecay;
	public int minibatchSize;
	public double momentum;
	public double regularization;
	
	public HyperParameters() {
		layerSizes = new int[] {784, 32, 10};
		activationType = Activation.ActivationType.Sigmoid;
		outputActivationType = Activation.ActivationType.Softmax;
		costType = Cost.CostType.CrossEntropy;
		trainingSplit = 0.8f;
		initialLearningRate = 0.05;
		learnRateDecay = 0.075;
		minibatchSize = 32;
		momentum = 0.9;
		regularization = 0.1;
	}
}
