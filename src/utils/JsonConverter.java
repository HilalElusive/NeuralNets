package utils;

import java.util.Arrays;

import functions.Activation;
import functions.Cost;
import utils.NetworkSnapshot.ConnectionSaveData;

public class JsonConverter {
	public static String serialize(NetworkSnapshot data) {
        StringBuilder jsonBuilder = new StringBuilder();
        
        // Start with layerSizes array
        jsonBuilder.append("{ \"layerSizes\": [");
        for (int i = 0; i < data.layerSizes.length; i++) {
            jsonBuilder.append(data.layerSizes[i]);
            if (i < data.layerSizes.length - 1) {
                jsonBuilder.append(", ");
            }
        }
        jsonBuilder.append("], ");

        // Add connections array
        jsonBuilder.append("\"connections\": [");
        for (int i = 0; i < data.connections.length; i++) {
            ConnectionSaveData conn = data.connections[i];
            jsonBuilder.append("{ \"weights\": [");
            for (int j = 0; j < conn.weights.length; j++) {
                jsonBuilder.append(conn.weights[j]);
                if (j < conn.weights.length - 1) {
                    jsonBuilder.append(", ");
                }
            }
            jsonBuilder.append("], \"biases\": [");
            for (int j = 0; j < conn.biases.length; j++) {
                jsonBuilder.append(conn.biases[j]);
                if (j < conn.biases.length - 1) {
                    jsonBuilder.append(", ");
                }
            }
            jsonBuilder.append("], \"activationType\": \"").append(conn.activationType).append("\" }");
            if (i < data.connections.length - 1) {
                jsonBuilder.append(", ");
            }
        }
        jsonBuilder.append("], ");

        // Add costFunctionType
        jsonBuilder.append("\"costFunctionType\": \"").append(data.costFunctionType).append("\" }");

        return jsonBuilder.toString();
    }
	
	public static NetworkSnapshot deserialize(String json) {
		NetworkSnapshot networkSaveData = new NetworkSnapshot();
	    // Remove the braces at the start and end
	    json = json.trim().substring(1, json.length() - 1);

	    // Find the start and end of the layerSizes array
	    int layerSizesStart = json.indexOf("[");
	    int layerSizesEnd = json.indexOf("]") + 1;  // Include the closing bracket

	    // Extract and parse layerSizes
	    String layerSizesStr = json.substring(layerSizesStart, layerSizesEnd);
	    String[] layerSizesParts = layerSizesStr.substring(1, layerSizesStr.length() - 1).split(",\\s*");
	    int[] layerSizes = new int[layerSizesParts.length];
	    for (int i = 0; i < layerSizesParts.length; i++) {
	    	layerSizes[i] = Integer.parseInt(layerSizesParts[i].trim());
	    }
	    networkSaveData.layerSizes = layerSizes;

	    // Find the start of the connections array
	    int connectionsStart = json.indexOf("\"connections\": [");
	    // Move the index to the start of the array
	    connectionsStart += "\"connections\": [".length();

	    // Find the end of the connections array by counting brackets
	    int bracketCount = 1; // We start after the opening bracket
	    int connectionsEnd = connectionsStart;
	    while (bracketCount > 0 && connectionsEnd < json.length()) {
	        char c = json.charAt(connectionsEnd);
	        if (c == '[') {
	            bracketCount++;
	        } else if (c == ']') {
	            bracketCount--;
	        }
	        connectionsEnd++;
	    }

	    // Extract the connections array string
	    String connectionsStr = json.substring(connectionsStart - 1, connectionsEnd);
	    // Split the connections array into individual connection JSON objects
	    String[] connectionParts = connectionsStr.substring(1, connectionsStr.length() - 1).split("}, \\{");
	    ConnectionSaveData[] connections = new ConnectionSaveData[connectionParts.length];

	    for (int i = 0; i < connectionParts.length; i++) {
	        String connStr = connectionParts[i].trim();
	        if (!connStr.startsWith("{")) {
	            connStr = "{" + connStr;
	        }
	        if (!connStr.endsWith("}")) {
	            connStr += "}";
	        }
	        connections[i] = parseConnection(connStr); // Parse each ConnectionSaveData object
	    }
	    networkSaveData.connections = connections;

	    // Parse costFunctionType
	    int costFunctionTypeStart = json.indexOf("\"costFunctionType\":") + "\"costFunctionType\":".length();
	    String costFunctionTypeStr = json.substring(costFunctionTypeStart).trim().replace("\"", "");
	    networkSaveData.costFunctionType = Cost.CostType.valueOf(costFunctionTypeStr);

	    return networkSaveData;
	}

	private static ConnectionSaveData parseConnection(String json) {
	    ConnectionSaveData connection = new ConnectionSaveData();

	    // Extract the weights array
	    int weightsStart = json.indexOf("\"weights\": [") + "\"weights\": [".length();
	    int weightsEnd = json.indexOf("]", weightsStart);
	    String weightsStr = json.substring(weightsStart, weightsEnd);
	    connection.weights = Arrays.stream(weightsStr.split(","))
	                    .map(String::trim)
	                    .mapToDouble(Double::parseDouble)
	                    .toArray();

	    // Extract the biases array
	    int biasesStart = json.indexOf("\"biases\": [", weightsEnd) + "\"biases\": [".length();
	    int biasesEnd = json.indexOf("]", biasesStart);
	    String biasesStr = json.substring(biasesStart, biasesEnd);
	    connection.biases = Arrays.stream(biasesStr.split(","))
	                   .map(String::trim)
	                   .mapToDouble(Double::parseDouble)
	                   .toArray();

	    // Extract the activation type
	    int activationTypeStart = json.indexOf("\"activationType\": \"", biasesEnd) + "\"activationType\": \"".length();
	    int activationTypeEnd = json.indexOf("\"", activationTypeStart);
	    String activationTypeStr = json.substring(activationTypeStart, activationTypeEnd);
	    connection.activationType = Activation.ActivationType.valueOf(activationTypeStr);

	    return connection;
	}


}
