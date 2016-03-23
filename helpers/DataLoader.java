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
	private static Map<String, Integer> userIndex;
	private static Map<String, Integer> itemIndex ;
	
	private static void loadIndices(String dir) throws IOException {
		userIndex = loadIndex(dir + "user_index.csv");
		System.out.println("loaded index of users");
		itemIndex = loadIndex(dir + "item_index.csv");
		System.out.println("loaded index of items");
	}
	
	private static Map<String, Integer> loadIndex(String fName) throws IOException {
		
		HashMap<String, Integer> index = new HashMap<String, Integer>();
		BufferedReader reader = new BufferedReader(new FileReader(fName));
		String line = reader.readLine();
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String id = fields[0];
			int ind = toJavaIndex(Integer.parseInt(fields[1]));
			index.put(id, ind);
			
		}
		reader.close();
		
		return index;
	}

	private static int toJavaIndex(int ind) {
		return ind - 1;
	}

	public static Dataset load(String dir, int numBrand, int splitIndex) throws IOException {
		
		loadIndices(dir);
		String rating_file = dir + splitIndex + "_split/" + "train_ratings.csv";
		RealMatrix ratings = loadRatings(rating_file);	// 
		RealMatrix edge_weights = loadEdgeWeights(dir + "edge_weights.csv");
		return new Dataset(ratings, edge_weights, numBrand);
	}

	// read edge weights from the file and fill in 0s for user pairs with no connection
	private static RealMatrix loadEdgeWeights(String fname) throws NumberFormatException, IOException {
		
		RealMatrix edge_weights = new Array2DRowRealMatrix(maxNumUser, maxNumUser);
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String vid = fields[1];
			double weight = Double.valueOf(fields[2]);
			
			int uIndex = userIndex.get(uid);
			int vIndex = userIndex.get(vid);
			edge_weights.setEntry(uIndex, vIndex, weight);
		}
		reader.close();
		int numUser = userIndex.size();
		edge_weights = edge_weights.getSubMatrix(0, numUser - 1, 0, numUser - 1);	// rm redundant rows and cols
		System.out.println("Loaded all edge weights.");

		return edge_weights;
	}
	
	private static RealMatrix loadRatings(String fname) throws IOException {
		RealMatrix ratings = new Array2DRowRealMatrix(maxNumUser, maxNumItem);
		
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String itemId = fields[1];
			double r = Double.valueOf(fields[2]);
			
			int uIndex = userIndex.get(uid);	// lookUpIndex(uid, userDict);
			int iIndex = itemIndex.get(itemId);
			ratings.setEntry(uIndex, iIndex, r);
		}
		
		reader.close();
		int numUser = userIndex.size();
		int numItem = itemIndex.size();
		System.out.println("numUser = " + numUser  + ", numItem = " + numItem);
		ratings = ratings.getSubMatrix(0, numUser - 1, 0, numItem - 1);		// rm redundant rows and cols
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
	@SuppressWarnings("unused")
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
