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
	
	private Map<String, Integer> userIndex;
	private Map<String, Integer> itemIndex ;
	private Map<String, Integer> brandIndex;
	
	public DataLoader(String dataDir) throws IOException {
		loadIndices(dataDir);
	}

	private void loadIndices(String dataDir) throws IOException {
		userIndex = loadIndex(dataDir + "user_index.csv");
		System.out.println("loaded index of " + userIndex.size() + " users");
		itemIndex = loadIndex(dataDir + "item_index.csv");
		System.out.println("loaded index of " + itemIndex.size() + " items");

		// XXX: on synthetic data, we do not load index of brands, smth is wrong here, tmp turn off this but need to handle later 
//		brandIndex = loadIndex(dataDir + "brand_index.tsv");
//		System.out.println("loaded index of " + brandIndex.size() + " brands");
	}
	
	private Map<String, Integer> loadIndex(String fName) throws IOException {
		
		HashMap<String, Integer> index = new HashMap<String, Integer>();
		BufferedReader reader = new BufferedReader(new FileReader(fName));
		
		int numFail2Parse = 0;
		String line = reader.readLine();
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",|\t");
			String id = fields[0];
			try {
				int normalIndex = Integer.parseInt(fields[1]);
				int ind = toJavaIndex(normalIndex);
				index.put(id, ind);
			} catch (NumberFormatException e) {
				//  handle exception
				numFail2Parse ++;
			}
			
		}
		reader.close();
		System.out.println("number of entries of which index cannot be parsed " + numFail2Parse);
		return index;
	}

	public Dataset load(String rating_file, String graph_file) throws IOException {
		
		RealMatrix ratings = loadRatings(rating_file);	// 
		
		RealMatrix edge_weights = loadEdgeWeights(graph_file);
//		int numBrand = brandIndex.size();
		//XXX: use this hard setting only for syn data
		int numBrand = 46; // 9K + 1
		return new Dataset(ratings, edge_weights, numBrand);
	}

	// read edge weights from the file and fill in 0s for user pairs with no connection
	private RealMatrix loadEdgeWeights(String fname) throws NumberFormatException, IOException {
		
//		System.out.println("loading edge weights...");
		int numUser = userIndex.size();
		RealMatrix edge_weights = new Array2DRowRealMatrix(numUser, numUser);
		BufferedReader reader = new BufferedReader(new FileReader(fname));
		String line = reader.readLine();	// skip header
		while ((line = reader.readLine()) != null) {
			String[] fields = line.split(",");
			String uid = fields[0];
			String vid = fields[1];
			double weight = Double.valueOf(fields[2]);
			
			if (inUserIndex(uid) && inUserIndex(vid)) {
				int uIndex = userIndex.get(uid);
				int vIndex = userIndex.get(vid);
				edge_weights.setEntry(uIndex, vIndex, weight);
			}
		}
		reader.close();

		return edge_weights;
	}

	/**
	 * read ratings from file and mark missing/NA ratings by -1
	 * @param fname
	 * @return
	 * @throws IOException
	 */
	public RealMatrix loadRatings(String fname) throws IOException {
		
//		System.out.println("loading ratings...");
		int numUser = userIndex.size();
		int numItem = itemIndex.size();
		RealMatrix ratings = new Array2DRowRealMatrix(numUser, numItem);
		ratings = ratings.scalarAdd(-1);	// mark missing ratings by -1
		
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

		return ratings;
	}

	private int toJavaIndex(int ind) {
		return ind - 1;
	}
	
	private boolean inUserIndex(String uid) {
		return userIndex.keySet().contains(uid);
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
	private  int lookUpIndex(String id, Map<String, Integer> id2Index) {
		
		if (id2Index.containsKey(id)) {
			return id2Index.get(id);
		} else {
			int size = id2Index.size() + 1;
			id2Index.put(id, size);
			return size;
		}
	}
}
