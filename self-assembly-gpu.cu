// for linux:
// nvcc self-assembly-gpu.cu -o temp -lcudart

#include <sys/time.h>
#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>

// ---- BODY PROPERTIES ---- //
#define N 6
#define MASS 1.0 // 0.0000000000131946891 //estimate with density 1.05g per cm cube
#define DIAMETER_PS 1.0 // Diameter of polystyrene spheres 1 micron
#define DIAMETER_NIPAM 0.08 // Diameter of polyNIPAM microgel particles 80 nanometers

// ---- FORCE FUNCTION ---- //
// Constant force for piecewise step function
#define MAX_ATTRACTION 10.0
#define REPULTION_MULTIPLIER 5.0
#define SHORT_RANGE_MULTIPLIER 0.01
#define LONG_RANGE_MULTIPLIER 0.5
#define LONG_RANGE_DISTANCE_MULTIPLIER 3.0
#define INITIAL_VELOCITY 0.5
#define DAMP .3
#define DT 0.0001

// ---- CONFIG PROPERTIES ---- //
// #define NUMBER_OF_RUNS 200
#define NUMBER_OF_THREADS 1 // this is the number of runs
#define MAX_TOTAL_KINETIC_ENERGY 0.002
#define INITIAL_SEPARATION 1.1
#define NO 0
#define YES 1

// ---- RESULTS PROPERTIES ---- //
float octa_count = 0.0;
float tetra_count = 0.0;
float other_count = 0.0;

double *OCTA_CPU, *TETRA_CPU, *OTHER_CPU;
double *OCTA_GPU, *TETRA_GPU, *OTHER_GPU;
float *RANDOM_POS_CPU, *RANDOM_VEL_CPU;
float *RANDOM_POS_GPU, *RANDOM_VEL_GPU;

// GPU setup
dim3 dimBlock;
dim3 dimGrid;

void set_up_cuda_devices()
{
    // number of threads per block will
    // always be maxed at 1024
    dimBlock.x = 1024;
    dimBlock.y = 1;
    dimBlock.z = 1;

    // figure out how many blocks you need
    if (NUMBER_OF_THREADS % dimBlock.x != 0) {
        dimGrid.x = NUMBER_OF_THREADS / dimBlock.x + 1;
    }
    else {
        dimGrid.x = NUMBER_OF_THREADS / dimBlock.x;
    }
    dimGrid.y = 1;
    dimGrid.z = 1;
    printf("Number of blocks: %i\n", dimGrid.x);
}

void allocate_memory()
{
    // Allocate Host (CPU) Memory with 0s
    OCTA_CPU = (double*)calloc(NUMBER_OF_THREADS, sizeof(double));
    TETRA_CPU = (double*)calloc(NUMBER_OF_THREADS, sizeof(double));
    OTHER_CPU = (double*)calloc(NUMBER_OF_THREADS, sizeof(double));
    RANDOM_POS_CPU = (float*)malloc(N*3*NUMBER_OF_THREADS*sizeof(float));
    RANDOM_VEL_CPU = (float*)malloc(N*3*NUMBER_OF_THREADS*sizeof(float));

    // generate random numbers for each experiment (flattened 3D array)
    // this includes random numbers for positions and velocities
	for (int i = 0; i < NUMBER_OF_THREADS; i++)
    {
        for (int j = 0; j < N*3; j++)
        {
            RANDOM_POS_CPU[N*3*i+j] = ((float)rand()/(float)RAND_MAX);
            RANDOM_VEL_CPU[N*3*i+j] = ((float)rand()/(float)RAND_MAX);
        }
    }

    // how to get forces for each body
    // for (int i=0; i < N*3*NUMBER_OF_THREADS; i+=3)
    // {
    //     printf("--------------------- random positions -----------------\n");
    //     printf("RANDOM_POS_CPU[%i]: %.15f\n", i, RANDOM_POS_CPU[i]);
    //     printf("RANDOM_POS_CPU[%i]: %.15f\n", i+1, RANDOM_POS_CPU[i+1]);
    //     printf("RANDOM_POS_CPU[%i]: %.15f\n", i+2, RANDOM_POS_CPU[i+2]);
    //     printf("--------------------- random velocities -----------------\n");
    //     printf("RANDOM_VEL_CPU[%i]: %.15f\n", i, RANDOM_VEL_CPU[i]);
    //     printf("RANDOM_VEL_CPU[%i]: %.15f\n", i+1, RANDOM_VEL_CPU[i+1]);
    //     printf("RANDOM_VEL_CPU[%i]: %.15f\n", i+2, RANDOM_VEL_CPU[i+2]);
    // }

    // Allocate Device (GPU) Memory and allocates the value of the specific pointer/array
    cudaMalloc(&OCTA_GPU, NUMBER_OF_THREADS*sizeof(double));
    cudaMalloc(&TETRA_GPU, NUMBER_OF_THREADS*sizeof(double));
    cudaMalloc(&OTHER_GPU, dimGrid.x*dimBlock.x*sizeof(double));
    cudaMalloc(&RANDOM_POS_GPU, N*3*NUMBER_OF_THREADS*sizeof(float));
    cudaMalloc(&RANDOM_VEL_GPU, N*3*NUMBER_OF_THREADS*sizeof(float));
}

// Cleaning up memory on both host and device after we are finished.
void clean_up(double *OCTA_CPU, double *TETRA_CPU, double *OTHER_CPU, double *OCTA_GPU, double *TETRA_GPU, double *OTHER_GPU)
{
    free(OCTA_CPU); free(TETRA_CPU); free(OTHER_CPU);
    cudaFree(OCTA_GPU); cudaFree(TETRA_GPU); cudaFree(OTHER_GPU);
}

// custom error function to make sure
// the GPU did what it was supposed to do
void errorCheck(const char *message)
{
  cudaError_t  error;
  error = cudaGetLastError();

  if(error != cudaSuccess)
  {
    printf("\n CUDA ERROR: %s - %s\n", message, cudaGetErrorString(error));
    exit(0);
  }
}

__global__ void self_assemble(double *OCTA_GPU, double *TETRA_GPU,
                              double *OTHER_GPU, float *RANDOM_POS_GPU,
                              float *RANDOM_VEL_GPU, int number_of_threads,
                              int n, unsigned int seed)
{
    // globals to store positions, velocities, and forces
    float p[6][3], v[6][3], f[6][3], mass[6];
    // OpenGL box size
    float xMin = -4.0;
    float xMax =  4.0;
    float yMin = -4.0;
    float yMax =  4.0;
    float zMin = -4.0;
    float zMax =  4.0;

    // each thread will perform an entire experiment
    int id = blockDim.x*blockIdx.x + threadIdx.x;
    if (id < number_of_threads) { // Make sure we do not go out of bounds
        // ------- SET INITIAL CONDITIONS ------- //
    	int i, j, ok_config;
    	float mag, distance, dx, dy, dz;
    	ok_config = NO;

    	while(ok_config == NO)
    	{
    		for (i = 0; i < n; i++)
    		{
    			// initialize mass of bodies
    			mass[i] = 1.0;
    			// intitialize positions
    			p[i][0] = RANDOM_POS_GPU[N*3*id+(i*3)] * (xMax - xMin) - ((xMax-xMin)/2);
    			p[i][1] = RANDOM_POS_GPU[N*3*id+(i*3+1)] * (yMax - yMin) - ((yMax-yMin)/2);
    			p[i][2] = RANDOM_POS_GPU[N*3*id+(i*3+2)] * (zMax - zMin) - ((yMax-yMin)/2);
    			// initialize velocities
    			mag = sqrt(p[i][0]*p[i][0]+p[i][1]*p[i][1]+p[i][2]*p[i][2]);
    			v[i][0] = INITIAL_VELOCITY*(-p[i][0]/mag)*RANDOM_VEL_GPU[N*3*id+(i*3)];
    			v[i][1] = INITIAL_VELOCITY*(-p[i][1]/mag)*RANDOM_VEL_GPU[N*3*id+(i*3+1)];
    			v[i][2] = INITIAL_VELOCITY*(-p[i][2]/mag)*RANDOM_VEL_GPU[N*3*id+(i*3+2)];
    		}
    		// make sure each body is a minimum distance from all the others
    		ok_config = YES;
    		for(i = 0; i < (n - 1); i++)
    		{
    			for(j = i + 1; j < n; j++)
    			{
    				dx = p[i][0]-p[j][0];
    				dy = p[i][1]-p[j][1];
    				dz = p[i][2]-p[j][2];
    				distance = sqrt(dx*dx + dy*dy + dz*dz);
    				// if(distance <= INITIAL_SEPARATION) {
    				// 	// printf("bodies too close!\n");
    				// 	ok_config = NO;
    				// }
    			}
    		}
    	}
        printf("initial positions:\n");
		for(i = 0; i < n; i++)
        {
            printf("p[%i][0]: %.15f\n", i, p[i][0]);
            printf("p[%i][1]: %.15f\n", i, p[i][1]);
            printf("p[%i][2]: %.15f\n", i, p[i][2]);
        }
        // -------------------------------------- //
		float total_kinetic_energy = 1.0;
        // printf("total kinetic energy: %.15f\n", total_kinetic_energy);
        // printf("DIAMETER_PS: %.15f\n", DIAMETER_PS);
		// stop updates when bodies have stopped moving
		int test = 0;
		while(total_kinetic_energy > MAX_TOTAL_KINETIC_ENERGY)
		{
            // ------- GET FORCES ------- //
        	float squared_distance;
        	float force_mag;

        	// initialize forces to 0
        	for (i = 0; i < n; i++)
        	{
        		f[i][0] = 0.0;
        		f[i][1] = 0.0;
        		f[i][2] = 0.0;
        	}
            // loop through every body
        	for (i = 0; i < n; i++)
        	{
        		// for each body, calculate distance and
        		// force from every other body
        		for (j = i+1; j < n; j++)
        		{
        			dx = p[j][0]-p[i][0];
        			dy = p[j][1]-p[i][1];
        			dz = p[j][2]-p[i][2];
                    // printf("p[%i][0]: %.15f\n", j, p[j][0]);
                    // printf("dx: %.15f\n", dx);
                    // printf("dy: %.15f\n", dy);
                    // printf("dz: %.15f\n", dz);
        			squared_distance = dx*dx + dy*dy + dz*dz;
        			distance = sqrt(squared_distance);
                    // printf("distance: %.15f\n", distance);

        			if (distance < DIAMETER_PS) // d < 1
        			{
        				force_mag = -REPULTION_MULTIPLIER*MAX_ATTRACTION; // -50
        			}
        			else if (distance < DIAMETER_PS + DIAMETER_NIPAM) // d < 1.08
        			{
        				force_mag =  MAX_ATTRACTION; //10
        			}
        			else if (distance < LONG_RANGE_DISTANCE_MULTIPLIER*DIAMETER_PS) // d < 3
        			{
        				force_mag =  MAX_ATTRACTION*SHORT_RANGE_MULTIPLIER; //.1
        			}
        			// make extra force that pulls to center
        			else force_mag = MAX_ATTRACTION*LONG_RANGE_MULTIPLIER; // 5
        			f[i][0] += force_mag*dx/distance;
        			f[j][0] -= force_mag*dx/distance;
        			f[i][1] += force_mag*dy/distance;
        			f[j][1] -= force_mag*dy/distance;
        			f[i][2] += force_mag*dz/distance;
        			f[j][2] -= force_mag*dz/distance;
        		}
        	}
            // printf("force mag: %.15f\n", force_mag);
    		// for(i = 0; i < n; i++)
            // {
            //     printf("f[%i][0]: %.15f\n", i, force_mag*dx/distance);
            //     printf("f[%i][1]: %.15f\n", i, f[i][1]);
            //     printf("f[%i][2]: %.15f\n", i, f[i][2]);
            // }
            // -------------------------------------- //
            // ------- UPDATE POSITIONS AND VELOCITIES ------- //
        	float dt = DT;
        	// update positions and velocities
        	for(i = 0; i < n; i++)
        	{
        		v[i][0] += ((f[i][0]-DAMP*v[i][0])/mass[i])*dt;
        		v[i][1] += ((f[i][1]-DAMP*v[i][1])/mass[i])*dt;
        		v[i][2] += ((f[i][2]-DAMP*v[i][2])/mass[i])*dt;

        		p[i][0] += v[i][0]*dt;
        		p[i][1] += v[i][1]*dt;
        		p[i][2] += v[i][2]*dt;
        	}
            // -------------------------------------- //
            printf("updated positions:\n");
    		for(i = 0; i < n; i++)
            {
                printf("p[%i][0]: %.15f\n", i, p[i][0]);
                printf("p[%i][1]: %.15f\n", i, p[i][1]);
                printf("p[%i][2]: %.15f\n", i, p[i][2]);
            }
            // ------- GET TOTAL KINETIC ENERGY ------- //
        	//calculate total kinetic energy
        	total_kinetic_energy = 0.0;
            // printf("total kinetic energy: %.15f\n", total_kinetic_energy);
        	for(i = 0; i < n; i++)
        	{
        		// total kinetic energy = summation{i=1}{n} 1/2 m v_i^2
        		total_kinetic_energy += 0.5*mass[i]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] +v[i][2]*v[i][2]);
        	}
            // -------------------------------------- //
            test = test + 1;
            if (test < 680169) {
                printf("total kinetic energy: %.15f\n", total_kinetic_energy);
                printf("count: %i\n", test);
            }
            else {
                break;
            }
            // if (count > 5) break;
		}
        // ------- IDENTIFY SHAPE ------- //
    	float squared_distance;

    	// figure out which shape is formed
    	float total_body_to_body_distance = 0.0;
    	for(i = 0; i < n - 1; i++)
    	{
    		for(j = i + 1; j < n; j++)
    		{
    			dx = p[j][0]-p[i][0];
    			dy = p[j][1]-p[i][1];
    			dz = p[j][2]-p[i][2];
    			squared_distance = dx*dx + dy*dy + dz*dz;
    			distance = sqrt(squared_distance);
    			total_body_to_body_distance += distance;
    		}
    	}
    	// theoretical distance: 16.2426
    	if(total_body_to_body_distance < 16.5426 && 15.9426 < total_body_to_body_distance)
    	{
            OCTA_GPU[id] = 1.0;
    	}
    	// theoretical distance: 17.168
    	else if(total_body_to_body_distance < 17.468 && 16.868 < total_body_to_body_distance)
    	{
            TETRA_GPU[id] = 1.0;
    	}
    	else
    	{
            OTHER_GPU[id] = 1.0;
    	}
        // -------------------------------------- //
    }
    // TETRA_GPU[id] = 5.0;
}

int main(int argc, char** argv)
{
	srand((unsigned int)time(NULL));
    printf("NUMBER_OF_THREADS = %i\n", NUMBER_OF_THREADS);
    // int i;
    timeval start, end;

    // Set the thread structure that you will be using on the GPU
    set_up_cuda_devices();

    // Partitioning off the memory that you will be using
    allocate_memory();

    //---- SELF-ASSEMBLY ON GPU ----//
    gettimeofday(&start, NULL);
    // Copy Memory from CPU to GPU
    cudaMemcpyAsync(OCTA_GPU, OCTA_CPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(TETRA_GPU, TETRA_CPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(OTHER_GPU, OTHER_CPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(RANDOM_POS_GPU, RANDOM_POS_CPU, N*3*NUMBER_OF_THREADS*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyAsync(RANDOM_VEL_GPU, RANDOM_VEL_CPU, N*3*NUMBER_OF_THREADS*sizeof(float), cudaMemcpyHostToDevice);
    errorCheck("error copying to GPU");

    // Calling the Kernel (GPU) function.
    self_assemble<<<dimGrid,dimBlock>>>(OCTA_GPU, TETRA_GPU, OTHER_GPU,
                                        RANDOM_POS_GPU, RANDOM_VEL_GPU,
                                        NUMBER_OF_THREADS, N, time(NULL));
    errorCheck("error calling GPU function");

    // Copy Memory from GPU to CPU
    cudaMemcpyAsync(OCTA_CPU, OCTA_GPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(TETRA_CPU, TETRA_GPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(OTHER_CPU, OTHER_GPU, NUMBER_OF_THREADS*sizeof(double), cudaMemcpyDeviceToHost);
    errorCheck("error copying to CPU");

    // TODO: sum up counts from GPU vars then calculate frequencies for the two
    // shapes
    for (int i = 0; i < NUMBER_OF_THREADS; i++) {
        printf("TETRA_CPU[%i]: %.15f\n", i, TETRA_CPU[i]);
    }
    float octa_count = 0.0;
    for (int i = 0; i < dimGrid.x; i++) {
        octa_count += OCTA_CPU[i * dimBlock.x];
    }
    printf("octa cpu count: %.15f\n", octa_count);
    float tetra_count = 0.0;
    for (int i = 0; i < dimGrid.x; i++) {
        tetra_count += TETRA_CPU[i * dimBlock.x];
    }
    printf("tetra cpu count: %.15f\n", tetra_count);
    float other_count = 0.0;
    for (int i = 0; i < dimGrid.x; i++) {
        other_count += OTHER_CPU[i * dimBlock.x];
    }
    printf("other cpu count: %.15f\n", other_count);

	// num_experiments = (float)run_count + 1.0;
	// octa_rate = octa_count/num_experiments;
	// tetra_rate = tetra_count/num_experiments;
	// other_rate = other_count/num_experiments;
	// printf("run count: %i\t octa_rate: %.2f\t tetra_rate: %.2f\t other_rate: %.2f\n",
	// 		(run_count+1), octa_rate, tetra_rate, other_rate);

    // add first entry in each block since blocks
    // can't communicate
    // dot_result = 0.0;
    // for (int i = 0; i < dimGrid.x; i++) {
    //     dot_result += OTHER_CPU[i * dimBlock.x];
    // }
    // printf("GPU dot product: %.15f\n", dot_result);
    //
    // // Stopping the timer
    // gettimeofday(&end, NULL);
    // // Calculating the total time used in the addition and converting it to milliseconds.
    // time = (end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec);
    // // Displaying the time
    // printf("Time in milliseconds= %.15f\n", (time/1000.0));
    //
    // // Displaying vector info you will want to comment out the vector print line when your
    // // vector becomes big. This is just to make sure everything is running correctly.
    // for(i = 0; i < N; i++)
    // {
    //     //printf("A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", i, OCTA_CPU[i], i, TETRA_CPU[i], i, OTHER_CPU[i]);
    // }

    // Displaying the last value of the addition for a check when all vector display has been commented out.
    //printf("Last Values are A[%d] = %.15f  B[%d] = %.15f  C[%d] = %.15f\n", N-1, OCTA_CPU[N-1], N-1, TETRA_CPU[N-1], N-1, OTHER_CPU[N-1]);

    // You're done so cleanup your mess.
    clean_up(OCTA_CPU, TETRA_CPU, OTHER_CPU, OCTA_GPU, TETRA_GPU, OTHER_GPU);

    return(0);
}

// OUTPUT
/*
Number of blocks: 8
CPU dot product: 41654167500.000000000000000
Time in milliseconds= 0.036000000000000

GPU dot product: 41654167500.000000000000000
Time in milliseconds= 0.144000000000000
*/
