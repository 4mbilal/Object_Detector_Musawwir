/************************************************************************/
/*                                                                      */
/*   kernel.h                                                           */
/*                                                                      */
/*   User defined kernel function. Feel free to plug in your own.       */
/*                                                                      */
/*   Copyright: Thorsten Joachims                                       */
/*   Date: 16.12.97                                                     */
/*                                                                      */
/************************************************************************/

/* KERNEL_PARM is defined in svm_common.h The field 'custom' is reserved for */
/* parameters of the user defined kernel. You can also access and use */
/* the parameters of the other kernels. Just replace the line 
             return((double)(1.0)); 
   with your own kernel. */

  /* Example: The following computes the polynomial kernel. sprod_ss
              computes the inner product between two sparse vectors. 

      return((CFLOAT)pow(kernel_parm->coef_lin*sprod_ss(a->words,b->words)
             +kernel_parm->coef_const,(double)kernel_parm->poly_degree)); 
  */

/* If you are implementing a kernel that is not based on a
   feature/value representation, you might want to make use of the
   field "userdefined" in SVECTOR. By default, this field will contain
   whatever string you put behind a # sign in the example file. So, if
   a line in your training file looks like

   -1 1:3 5:6 #abcdefg

   then the SVECTOR field "words" will contain the vector 1:3 5:6, and
   "userdefined" will contain the string "abcdefg". */

double custom_kernel(KERNEL_PARM *kernel_parm, SVECTOR *a, SVECTOR *b) 
     /* plug in you favorite kernel */                          
{
	static double max;
	int x;
	static int hist[10];
	register double sum=0;
	double temp;
    register mWORD *ai,*bj;
    ai=a->words;
    bj=b->words;
    while (ai->wnum && bj->wnum) {
      if(ai->wnum > bj->wnum) {
	bj++;
      }
      else if (ai->wnum < bj->wnum) {
	ai++;
      }
      else {


		  //sum+=(ai->weight) * (bj->weight);		//linear kernel

		  //SSE kernel
		 /* temp = ai->weight - bj->weight;
		  temp = temp*temp;
		  sum += 512 - sqrt(temp);
		  */
		 
		  //AD kernel
		 
		 // sum += 64-abs((ai->weight) - (bj->weight));										

			
		  //HIK
		  if((ai->weight) < (bj->weight))
			  sum+=(ai->weight);
		  else
			  sum+=(bj->weight);
			
		
		//printf("\nA = %f, B = %f, Sum = %f",(ai->weight),(bj->weight),sum);

		 // sum += (ai->weight) * (bj->weight);
			  
		 
	ai++;
	bj++;
      }
    }
//	if(sum!=511)
//	if(sum!=511*2)
	//	printf("\nSum = %f",sum);
	//SSE Kernel
	//sum = 723-sqrt(sum);

	//RBF
	//sum = exp(0.5*sqrt(sum));
	//sum =   exp(-0.05*(a->twonorm_sq - 2 * sprod_ss(a, b) + b->twonorm_sq)));

	/*
	printf("\nSum = %f",sum);
	hist[(int)((sum/10)*10)] = hist[(int)((sum/10)*10)] + 1;
	if(sum>max){
		max = sum;
		//printf("\t%f",sum);
	}
//	printf("\t%f",max);
	printf("\n\n");
	for(x=0;x<10;x++)
		printf("%d, ",hist[x]);
		
	getchar();
	*/
	//getchar();
    return((double)sum);
//  return((double)(1.0)); 
}
