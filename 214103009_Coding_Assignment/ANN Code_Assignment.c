// Defining library
#include<stdio.h>
#include<conio.h>
#include<stdlib.h>
#include<math.h>


int main()
{
	float MSE;
    FILE *in;
    FILE *out;
    FILE *gra;
    in = fopen("input.txt","r");
    out = fopen("output.txt","w");
    gra = fopen("Graph2.txt","w");
    
    
    // Defining the variables
	int L,M,N,P,TP;
	float eta, alp,numt;
	int p,count=0;
    
	// Taking values from input file
	fscanf(in,"%d",&L);
	fscanf(in,"%d",&M);
	fscanf(in,"%d",&N);
	fscanf(in,"%d",&P);
	fscanf(in,"%f",&eta);
	fscanf(in,"%d",&TP);
	
	
	//Defining Input and Weight matrices
	float I[L+1][P+1],V[L+1][M+1],W[M+1][N+1];
	//Defining the target outputs for each pattern
	float TO[N+1][P+1];
	float NOT[N][P];
	//Taking values for input matrix from input text file
	int i,j,k;
	for(i=1;i<=P;i++)
	{
		for(j=0;j<=L;j++)
		{
			if(j==0)
			{
				I[j][i] = 1;
			}
			else
			{
				fscanf(in,"%f",&I[j][i]);
			}
		}
	}
	
	/*for(i=1;i<=P;i++)
	{
		for(j=0;j<=L;j++)
		{
			printf("%f\t",I[j][i]);
		}
		printf("\n");
	}*/
	
	//Taking random values for V and W weight matrices
	double randf;
	//for matrix V
	for(i=0;i<=L;i++)
	{
		for(j=1;j<=M;j++)
		{
			if(i==0)
			{
				V[i][j] = 0;
			}
			else
			{
				numt = rand()%10;
				V[i][j] = numt/10;
			}
		}
	}
	
	//now for matrix W
		for(i=0;i<=M;i++)
	{
		for(j=1;j<=N;j++)
		{
			if(i==0)
			{
				W[i][j] = 0;
			}
			else
			{
				numt = rand()%10;
				W[i][j] = numt/10;
			}
		}
	}
	
	//Taking target output matrix from text file
	for(p=1;p<=P;p++)
	{
		for(k=1;k<=N;k++)
		{
			//printf("TO[%d][%d] = ",k,p);
			fscanf(in,"%f",&TO[k][p]);
		}
	}
	
	//Normalisation of Input and output data
	//first finding max and min values
	float Max[L+1],Min[L+1];
	for(i=1;i<=L;i++)
	{
		Max[i] = I[i][1];
		Min[i] = I[i][1];
		for(p=1;p<=P;p++)
		{
			if(I[i][p]>Max[i])
			{
				Max[i] = I[i][p];
			}
			if(I[i][p]<Min[i])
			{
				Min[i] = I[i][p];
			}
		}
	}
	
	for(i=1;i<=L;i++)
	{
	    for(p=1;p<=P;p++)
	    {
	        I[i][p] = 0.1 + (0.8*((I[i][p] - Min[i])/(Max[i] - Min[i])));
	    }
	}
	
    float MaxT[N+1],MinT[N+1];
	for(k=1;k<=N;k++)
    {
    	MaxT[k] = TO[k][1];
    	MinT[k] = TO[k][1];
    	for(p=1;p<=P;p++)
    	{
    		if(TO[k][p]>MaxT[k])
    		{
    			MaxT[k] = TO[k][p];
    		}
    		if(TO[k][p]<MinT[k])
    		{
    			MinT[k] = TO[k][p];
    		}
    	}
    }
    	
    	
    for(k=1;k<=N;k++)
    {
        for(p=1;p<=P;p++)
        {
            TO[k][p] = 0.1 + (0.8*((TO[k][p] - MinT[k])/(MaxT[k] - MinT[k])));
        }
    }
	
	//defining the IH matrix(input to the hidden neuron)
	float IH[M+1][P+1],OH[M+1][P+1];
	
	//defining the input and output matrix for output layer
	float IO[N+1][P+1],OO[N+1][P+1];
	
	// defining the delta values
	float delW[M+1][N+1],debydw[M+1][N+1];
	
	// defining required values
	float delV[L+1][M+1],dedv1[L+1][M+1],dedv2[L+1][M+1];
	
	//Using loop
	do
	{
		// Now starting forward pass calculations
		//calculation for input to hidden neuron for each pattern
		for(p=1;p<=P;p++)
		{
			for(j=1;j<=M;j++)
			{
			    IH[j][p] = 0;
				for(i=0;i<=L;i++)
				{
				
					IH[j][p] = IH[j][p] + (I[i][p]*V[i][j]);
				}

				// calculating output of hidden neuron
				float temp = 0;
				temp = exp(-IH[j][p]);
				OH[j][p] = (1/(1+temp));
			}
		}

		//bias for hidden neuron defined as 1
		for(p=1;p<=P;p++)
		{
			OH[0][p] = 1;
		}

		//now solving for output layer
		//calculation for input to output neuron for each pattern
		for(p=1;p<=P;p++)
		{
			for(k=1;k<=N;k++)
			{
			    IO[k][p] = 0;
				for(j=0;j<=M;j++)
				{
					IO[k][p] = IO[k][p] + (OH[j][p]*W[j][k]);
				}

				// calculating output of output neuron
				float t1 = 0, t2 = 0;
				//temp = exp((-1)*(IH[j][p]));
				t1 = exp(IO[k][p]);
				t2 = exp((-1)*(IO[k][p]));
				OO[k][p] = ((t1 - t2)/(t1 + t2));
			}
		}
		
		for(p=1;p<=P;p++)
		{
			for(k=1;k<=N;k++)
			{
				//printf("%f\n",OO[k][p]);
			}
		}
		
		//Calculating the mean square error for given training patterns
		//printf("\n Mean sqaure error MSE is as follows\n");
		float sum = 0;
		MSE = 0;
		for(p=1;p<=P;p++)
		{
			for(k=1;k<=N;k++)
			{
				sum = sum + ((TO[k][p]-OO[k][p])*(TO[k][p]-OO[k][p])*0.5);
			}
		}
		
		//printf("%f",sum);

		MSE = sum/P;
		printf("MSE = %f\n",MSE);
		fprintf(gra,"%f\n",MSE);
		//fprintf(out," The mean sqaure errpor is = %f\n",MSE);
		//fprintf(out,"%f\n",MSE);

		//updating the "W" weight values

		for(j=0;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
			    debydw[j][k] = 0;
				for(p=1;p<=P;p++)
				{
					float dd1,dd2,dd2t1,dd2t2,dd2t3,dd3;
					dd1 = -(TO[k][p]-OO[k][p]);
					dd2t1 = exp(IO[k][p]);
					dd2t2 = exp(-IO[k][p]);
					dd2t3 = dd2t1+dd2t2;
					dd2 = 4/(dd2t3*dd2t3);
					dd3 = OH[j][p];
	 				debydw[j][k] = debydw[j][k] + (dd1*dd2*dd3);
				}
				//calculating delta w
				//printf("\ndebydw for %d is %f\n",k,debydw[j][k]);
				delW[j][k] = ((-eta)/P)*debydw[j][k];
			}
		}
		//now printing the delta w values
		/*printf("\nbelow are the delta w values\n");
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				printf("delW[%d][%d] = %f\n",j,k,delW[j][k]);
			}
		}*/

		//now updating V matrix values by taking avg error from the output layer

		for(i=0;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
			    dedv1[i][j] = 0;
			    //dedv2[i][j] = 0;
				for(p=1;p<=P;p++)
				{
					for(k=1;k<=N;k++)
					{
						float dv1,dv2,dv3,dv4,dv5;
						dv1 = -(TO[k][p]-OO[k][p]);
						dv2 = 4/((exp(IO[k][p])+exp(-IO[k][p]))*(exp(IO[k][p])+exp(-IO[k][p])));
						dv3 = W[j][k];
						//dv4 = OH[j][p]*(1-OH[j][p]);
						dv4 = (exp(-IH[j][p]))/((1+exp(-IH[j][p]))*(1+exp(-IH[j][p])));
						dv5 = I[i][p];
						dedv1[i][j] = dedv1[i][j] + (dv1*dv2*dv3*dv4*dv5);
						//printf("dedv1 value = %f\n",dedv1[i][j]);
					}
					//dedv2[i][j] = dedv2[i][j] + dedv1[i][j];
					//printf("dedv2 value = %f\n",dedv2[i][j]);
				}
				//printf("dedv1 value = %f\n",dedv1[i][j]);
				delV[i][j] = (-eta)*(1/((float)N*(float)P))*(dedv1[i][j]);
				//printf("delVvalue = %f\n",delV[i][j]);
			}
		}
		//printf("%f",delV[1][1]);
		//now printing delV values
		//printf("\nbelow are the delta V values\n");
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				//printf("delV[%d][%d] = %f\n",i,j,delV[i][j]);
			}
		}

		//now updating and printing V and W values
		//printing W values
		//printf("\nFollowing are the updated W values\n");
		for(j=1;j<=M;j++)
		{
			for(k=1;k<=N;k++)
			{
				W[j][k] = W[j][k] + delW[j][k];
				//printf("W[%d][%d] = %f\n",j,k,W[j][k]);
			}
		}

		//printing V values
		//printf("\nFollowing are the updated V values\n");
		for(i=1;i<=L;i++)
		{
			for(j=1;j<=M;j++)
			{
				V[i][j] = V[i][j] + delV[i][j];
				//printf("V[%d][%d] = %f\n",i,j,V[i][j]);
			}
		}
		
		//fprintf(out,"%f\n",V[1][1]);
		count++;
		//fprintf(gra,"%d\n",count);

	}while(MSE>0.001&&count<110000);
	fprintf(out,"MSE = %f\n",MSE);
	fprintf(out,"No of Iteration = %d",count);
	
    //now getting output for testing patterns
    //inputs from testing pattern
	for(i=1;i<=TP;i++)
	{
		for(j=0;j<=L;j++)
		{
			if(j==0)
			{
				I[j][i] = 1;
			}
			else
			{
				fscanf(in,"%f",&I[j][i]);
			}
		}
	}
    
	//Taking target output matrix from text file of testing pattern
	for(p=1;p<=TP;p++)
	{
		for(k=1;k<=N;k++)
		{
			fscanf(in,"%f",&TO[k][p]);
		}
	}

	//Normalisation of Input and output data of testing pattern
	//first finding max and min values
	float MaxTP[L+1],MinTP[L+1];
	for(i=1;i<=L;i++)
	{
		MaxTP[i] = I[i][1];
		MinTP[i] = I[i][1];
		for(p=1;p<=TP;p++)
		{
			if(I[i][p]>MaxTP[i])
			{
				MaxTP[i] = I[i][p];
			}
			if(I[i][p]<MinTP[i])
			{
				MinTP[i] = I[i][p];
			}
		}
	}
	
	for(i=1;i<=L;i++)
	{
	    for(p=1;p<=TP;p++)
	    {
	        I[i][p] = 0.1 + (0.8*((I[i][p] - MinTP[i])/(MaxTP[i] - MinTP[i])));
	    }
	}
	
    float MaxTTP[N+1],MinTTP[N+1];
    for(k=1;k<=N;k++)
    {
    	MaxTTP[k] = TO[k][1];
    	MinTTP[k] = TO[k][1];
    	for(p=1;p<=TP;p++)
    	{
    		if(TO[k][p]>MaxTTP[k])
    		{
    			MaxTTP[k] = TO[k][p];
    		}
    		if(TO[k][p]<MinTTP[k])
    		{
    			MinTTP[k] = TO[k][p];
    		}
    	}
    }
    	
    	
    for(k=1;k<=N;k++)
    {
        for(p=1;p<=TP;p++)
        {
            TO[k][p] = 0.1 + (0.8*((TO[k][p] - MinTTP[k])/(MaxTTP[k] - MinTTP[k])));
        }
    }
	
		// Now starting forward pass calculations for testing pattern
		//calculation for input to hidden neuron for each pattern
		for(p=1;p<=TP;p++)
		{
			for(j=1;j<=M;j++)
			{
			    IH[j][p] = 0;
				for(i=0;i<=L;i++)
				{
				
					IH[j][p] = IH[j][p] + (I[i][p]*V[i][j]);
				}

				// calculating output of hidden neuron
				float temp = 0;
				temp = exp(-IH[j][p]);
				OH[j][p] = (1/(1+temp));
			}
		}

		//bias for hidden neuron defined as 1
		for(p=1;p<=TP;p++)
		{
			OH[0][p] = 1;
		}

		//now solving for output layer
		//calculation for input to output neuron for each pattern
		for(p=1;p<=TP;p++)
		{
			for(k=1;k<=N;k++)
			{
			    IO[k][p] = 0;
				for(j=0;j<=M;j++)
				{
					IO[k][p] = IO[k][p] + (OH[j][p]*W[j][k]);
				}

				// calculating output of output neuron
				float t1 = 0, t2 = 0;
				//temp = exp((-1)*(IH[j][p]));
				t1 = exp(IO[k][p]);
				t2 = exp((-1)*(IO[k][p]));
				OO[k][p] = ((t1 - t2)/(t1 + t2));
			}
		}
		
		//Output of training pattern
		for(p=1;p<=TP;p++)
		{
			for(k=1;k<=N;k++)
			{
				printf("%f\n",OO[k][p]);
			}
		}
	
		
		//Calculating the mean square error for given testing patterns
		//printf("\n Mean sqaure error MSE is as follows\n");
		float sumT = 0,MSET;
		MSET = 0;
		for(p=1;p<=TP;p++)
		{
			for(k=1;k<=N;k++)
			{
				sumT = sumT + ((TO[k][p]-OO[k][p])*(TO[k][p]-OO[k][p])*0.5);
			}
		}

		MSET = sumT/TP;
		fprintf(out," \nThe mean sqaure error of testing pattern is = %f\n",MSET);
		
	//Denormalisation of output
	fprintf(out,"\nDenormalised output is as follows\n");
	for(p=1;p<=TP;p++)
	{
	    for(k=1;k<=N;k++)
	    {
	        NOT[k][p] = (((OO[k][p] - 0.1) * (MaxTTP[k] - MinTTP[k])) / 0.8) + MinTTP[k];
	        fprintf(out,"%f\n",NOT[k][p]);
	    }
	}
		
		

	
	return 0;

}

