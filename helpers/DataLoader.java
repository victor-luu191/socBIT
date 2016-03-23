package helpers;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.apache.commons.math3.linear.Array2DRowRealMatrix;
import org.apache.commons.math3.linear.RealMatrix;

import defs.Dataset;

public class DataLoader {
	
	private static int maxNumUser = 2000;
	private static int maxNumItem = 2000;
	
	public static Dataset load(String dir, int numBrand) throws IOException {
		
		RealMatrix ratings = loadRatings(dir + "ratings.csv");	// train_ratings.csv
		RealMatrix edge_weights = loadEdgeWeights(dir + "edge_weights.csv");
		return new Dataset(ratings, edge_weights, numBrand);
	}

	// read edge weights from the file and fill in 0s for user pairs with no connection
	private static RealMatrix loadEdgeWeights(String fname) throws NumberFormatException, IOException {
		
		RealMatrix edge_weights = new Array2DRowRealMatrix(maxNumUser, maxNumUser);
		Map<String, Integer> userMap = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String vid = fields[1];
			double weight = Double.valueOf(fields[2]);
			
			int uIndex = lookUpIndex(uid, userMap);
			int vIndex = lookUpIndex(vid, userMap);
			edge_weights.setEntry(uIndex, vIndex, weight);
		}
		reader.close();
		int numUser = userMap.size();
		edge_weights = edge_weights.getSubMatrix(1, numUser, 1, numUser);	// rm redundant rows and cols
		System.out.println("Loaded all edge weights.");

		return edge_weights;
	}
	
	private static RealMatrix loadRatings(String fname) throws IOException {
		RealMatrix ratings = new Array2DRowRealMatrix(maxNumUser, maxNumItem);
		
		Map<String, Integer> itemMap = new HashMap<String, Integer>();
		Map<String, Integer> userMap = new HashMap<String, Integer>();
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String itemId = fields[1];
			double r = Double.valueOf(fields[2]);
			
			int userIndex = lookUpIndex(uid, userMap);
			int itemIndex = lookUpIndex(itemId, itemMap);
			ratings.setEntry(userIndex, itemIndex, r);
		}
		
		reader.close();
		int numUser = userMap.size();
		int numItem = itemMap.size();
		System.out.println("numUser = " + numUser  + ", numItem = " + numItem);
		ratings = ratings.getSubMatrix(1, numUser, 1, numItem);		// rm redundant rows and cols
		System.out.println("Loaded all ratings.");
//		System.out.println("First row of ratings: ");
//		System.out.println(ratings.getRowVector(0).toString());
//		System.out.println();
		return ratings;
	}

	/**
	 * Look up the index corresponding to the specified {@link id} if {@link id} can be found in the map {@link id2Index}, 
	 * Otherwise add the {@link id} to the map and increment the map {@link size} 
	 * @param id
	 * @param id2Index
	 * @param size
	 * @return
	 */
	private static int lookUpIndex(String id, Map<String, Integer> id2Index) {
		
		if (id2Index.containsKey(id)) {
			return id2Index.get(id);
		} else {
			int size = id2Index.size() + 1;
			id2Index.put(id, size);
			return size;
		}
	}
}
