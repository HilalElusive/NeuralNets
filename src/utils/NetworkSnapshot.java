package utils;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

import core.NeuralNetwork;
import functions.Activation;
import functions.Cost;

public class NetworkSnapshot {
	public int[] layerSizes;
    public ConnectionSaveData[] connections;
    public Cost.CostType costFunctionType;

    // Load network from saved data
    public NeuralNetwork loadNetwork() {
    	NeuralNetwork network = new NeuralNetwork(layerSizes);
    	
        for (int i = 0; i < connections.length; i++) {
            ConnectionSaveData loadedConnection = connections[i];

            System.arraycopy(loadedConnection.weights, 0, network.getLayers()[i].getWeights(), 0, loadedConnection.weights.length);
            System.arraycopy(loadedConnection.biases, 0, network.getLayers()[i].getBiases(), 0, loadedConnection.biases.length);
            network.getLayers()[i].setActivationFunction(Activation.getActivationFromType(loadedConnection.activationType));
        }
        network.setCostFunction(Cost.getCostFromType(costFunctionType));
        return network;
    }

    // Load save data from file
    public static NeuralNetwork loadNetworkFromFile(String path) throws IOException {
        try (BufferedReader reader = new BufferedReader(new FileReader(path))) {
            StringBuilder data = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                data.append(line);
            }
            return loadNetworkFromData(data.toString());
        }
    }

    public static NeuralNetwork loadNetworkFromData(String loadedData) {
        NetworkSnapshot saveData = JsonConverter.deserialize(loadedData);

        return saveData.loadNetwork();
    }

    public static String serializeNetwork(NeuralNetwork network) {
    	NetworkSnapshot saveData = new NetworkSnapshot();
        saveData.layerSizes = network.getLayerSizes();
        saveData.connections = new ConnectionSaveData[network.getLayers().length];
        saveData.costFunctionType = network.getCostFunction().costFunctionType();

        for (int i = 0; i < network.getLayers().length; i++) {
            saveData.connections[i] = new ConnectionSaveData();
            saveData.connections[i].weights = network.getLayers()[i].getWeights();
            saveData.connections[i].biases = network.getLayers()[i].getBiases();
            saveData.connections[i].activationType = network.getLayers()[i].getActivation().getActivationType();
        }

        return JsonConverter.serialize(saveData);
    }

    public static void saveToFile(String networkSaveString, String path) throws IOException {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(path))) {
            writer.write(networkSaveString);
        }
    }

    public static void saveToFile(NeuralNetwork network, String path) throws IOException {
        saveToFile(serializeNetwork(network), path);
    }

    public static class ConnectionSaveData {
        public double[] weights;
        public double[] biases;
        public Activation.ActivationType activationType;
    }

}
