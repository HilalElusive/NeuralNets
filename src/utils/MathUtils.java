package utils;
import java.util.Random;

public class MathUtils {
	private static final Random random = new Random();

    public static double[][] randomMatrix(int rows, int cols, double scale) {
        double[][] matrix = new double[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = (2 * random.nextDouble() - 1) * scale;
            }
        }
        return matrix;
    }
    
	public static double[][] matrixMul(double[][] A, double[][] B) {
		int aRows = A.length;
		int aColumns = A[0].length;
		int bRows = B.length;
		int bColumns = B[0].length;

        if (aColumns != bRows)
            throw new IllegalArgumentException("A:Rows: " + aColumns + " did not match B:Columns " + bRows + ".");
     
        double[][] C = new double[aRows][bColumns];
        for (int i = 0; i < aRows; i++) {
            for (int j = 0; j < bColumns; j++) {
                C[i][j] = 0.00000;
            }
        }
        for (int i = 0; i < aRows; i++) { // aRow
            for (int k = 0; k < aColumns; k++) { // aColumn
                for (int j = 0; j < bColumns; j++) { // bColumn
                    C[i][j] += A[i][k] * B[k][j];
                }
            }
        }
        return C;
	}
	
	public static void printMatrix(double[][] matrix) {
        for (double[] row : matrix) {
            for (double element : row) {
            	System.out.print(String.format("%12.7e ", element));
            }
            System.out.println();
        }
    }
}
