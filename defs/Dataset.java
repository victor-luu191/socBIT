package defs;

import org.apache.commons.math3.linear.NonSquareMatrixException;
import org.apache.commons.math3.linear.RealMatrix;

public class Dataset {
	
	public RealMatrix ratings;		// user-item
	public RealMatrix edge_weights;	// user-user square matrix, for any two unconnected users the weight is 0
	
	// derived fields
	public int numUser;
	public int numItem;
	public int numBrand;
	
	/**
	 * Precond: {@code edge_weight} is a square matrix and {@code nrow(edge_weight) = nrow(rating)} 
	 * @param ratings
	 * @param edge_weights
	 */
	public Dataset(RealMatrix ratings, RealMatrix edge_weights) {
		
		super();
		
		if (! edge_weights.isSquare()) {
			System.out.println("the edge weight matrix is NOT square!!!");
			throw new NonSquareMatrixException(edge_weights.getColumnDimension(), edge_weights.getRowDimension());
		} else {
			if (edge_weights.getRowDimension() != ratings.getRowDimension()) {
				System.out.println("The edge weight matrix and the rating matrix must have the same number of rows "
									+ "(the number of users)");
			} else {
				this.ratings = ratings;
				this.edge_weights = edge_weights;
				numUser = ratings.getRowDimension();
				numItem = ratings.getColumnDimension();
			}
		}
	}
	
	
}
