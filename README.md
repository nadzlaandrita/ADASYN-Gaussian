# ADASYN-Gaussian
This notebook contains the raw code for the latest augmentation method of the modified ADASYN algorithm, ADASYN-Gaussian.

**[Algorithm of ADASYN-Gaussian]**

**Input:**

	Training dataset D_tr with m samples {(x_i,y_i )},i=1,…,m, where x_i∈R^n represents an instance in an n-dimensional feature space, and y_i∈Y is the corresponding class label.
	
	Let:
	
	m_s : number of minority class samples
	
	m_l : number of majority class samples
	
	Such that m_s  ≤ m_l and m= m_s+ m_l

	Parameters:
	K : number of nearest neighbors
	β∈[0,1] : desired balance level
	d_th ; imbalance threshold
	ϵ : small constant for regularization

**Procedure:**

(1)	Compute the degree of class imbalance: 

													  d=  m_s/m_l                            (1)
    
	where d∈(0,1]

(2)	If d≤d_th proceed with oversampling:

	  a. Compute the total number of synthetic samples:
      
													  G=(m_l- m_s )  × β                    (2)  
        
		where β controls the desired balance level. When β=1, a fully balanced dataset is obtained.
	  
	  b. Determine the number of synthetic samples per minority instances:
      
													  g_i=max⁡(⌊G/m_s ⌋,1)               (3)
	  
	  c. For each minority instance x_i∈minorityclass:
	  
		  i. Find the K nearest neighbors of x_i using Euclidean distance.
	      
		  ii. Select minority-class neighbors:
          
										N_i={x_j│x_j∈KNN(x_i ),〖 y〗_j=minorityclass}              (4)
	      
		  iii. If N_i≠∅, repeat g_i times:
	      
			  • Construct the local sample set:
              
														S_i= N_i∪{x_i }           (5)
	         
			  • Estimate the local mean:
              
												μ_i=  1/(|S_i |) ∑_(x_j∈S_i)▒x_j               (6)
	          
			  • Estimate the covariance matrix:
              
									Σ_i=  1/(|S_i |-1) ∑_(x_j∈S_i)▒〖(x_j-μ_i)(x_j- μ_i)〗^T             (7)
	          
			  • Apply covariance regularization to ensure positive semi-definiteness:
              
			  Symmetrization:
              
														Σ_i=  (Σ_i+ Σ_i^T)/2          (8)
              
			  Diagonal loading:
              
															Σ_i= Σ_i+ ϵΙ     (9)
              
			  Eigenvalue correction:
              
															Σ_i=〖QΛQ〗^T       (10)
                                                            
															  Λ=max⁡(Λ,ϵ)      (11)
	          
			  • Generate a synthetic sample from the multivariate Gaussian distribution:
              
														  x_new  ~ ϰ(μ_i,Σ_i )      (12)
	
	(3) Aggregate all generated samples and combine with the original dataset:
    
														D^'=D_tr  ∪ D_synthetic                (13)
	
	(4) If d≥d_th, no oversampling is performed.


**Output:**

A new dataset:

D^'=〖{(x_i,y_i )}〗_(i=1)^m'

where  m^'>m, with an improved class balanced
