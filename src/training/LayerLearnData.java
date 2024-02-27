package training;

import core.Layer;

public class LayerLearnData {
	public double[] inputs;
	public double[] weightedInputs;
	public double[] activations;
	public double[] nodeValues;

	public LayerLearnData(Layer layer)
	{
		weightedInputs = new double[layer.getNumNodesOut()];
		activations = new double[layer.getNumNodesOut()];
		nodeValues = new double[layer.getNumNodesOut()];
	}
}
