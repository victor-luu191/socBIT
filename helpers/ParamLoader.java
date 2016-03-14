package helpers;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import defs.Params;
import defs.SocBIT_Params;

public class ParamLoader {
	
	private static final int maxDim = 2000;
	
	public static Params load(String gtParamsDir) throws IOException {
		System.out.println("Loading gt params from folder " + gtParamsDir);
		
		double[] decPrefs = loadDecPref(gtParamsDir);
		RealMatrix topicUser = loadMat(gtParamsDir, "topic_user_feats.csv");
		RealMatrix brandUser = loadMat(gtParamsDir, "brand_user_feats.csv");
		RealMatrix topicItem = loadMat(gtParamsDir, "topic_item_feats.csv");
		RealMatrix brandItem = loadMat(gtParamsDir, "brand_item_feats.csv");
		
		int numTopic = topicItem.getRowDimension();
		int numUser = topicUser.getColumnDimension();
		int numBrand = brandUser.getRowDimension();
		int numItem = topicItem.getColumnDimension();
		System.out.println("numTopic, numUser, numBrand, numItem");
		System.out.println(numTopic + "," + numUser + "," + numBrand + "," + numItem);
		
		return new SocBIT_Params(decPrefs, topicUser, brandUser, topicItem, brandItem);
	}

	private static double[] loadDecPref(String gtParamsDir) throws FileNotFoundException, IOException {
		
		String fname = gtParamsDir + "/decision_pref.csv";
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		
		List<Double> decPrefs = new ArrayList<Double>();
		String line = reader.readLine();
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
//			String uid = fields[0];
			decPrefs.add(Double.parseDouble(fields[1]));
		}
		reader.close();
		double[] prefs = new double[decPrefs.size()];
		for (int u = 0; u < prefs.length; u++) {
			prefs[u] = decPrefs.get(u);
		}
		return prefs;
	}
	
	private static RealMatrix loadMat(String gtParamsDir, String fname) throws IOException {
		
		RealMatrix matrix = new Array2DRowRealMatrix(maxDim, maxDim);
		BufferedReader reader = new BufferedReader(new FileReader(gtParamsDir + fname));
		
		int numRow = 0;
		
		String line = reader.readLine();
		int numCol = line.split(",").length - 1;	// bc the first column contains row indices not values
		while ((line = reader.readLine()) != null) {
			numRow ++;
			String[] fields = line.split(",");
			int row = Integer.parseInt(fields[0].replace("\"", ""));
			for (int col = 1; col < fields.length; col++) {
				matrix.setEntry(row, col, Double.parseDouble(fields[col]));
			}
		}
		matrix = matrix.getSubMatrix(1, numRow, 1, numCol);
		reader.close();
		return matrix;
	}
}
