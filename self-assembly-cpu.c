// General CPU code. Run on the upper trianglar part of the force matrix.
// for linux:
// gcc self-assembly-cpu.c -o temp -lglut -lm -lGLU -lGL
// for mac:
// nvcc self-assembly-cpu.c -o temp -Xlinker -framework -Xlinker OpenGL -lglut -lm -lGLEW
//#define XWindowSize 700
//#define YWindowSize 700
// OR
// gcc self-assembly-cpu.c -o temp -L/System/Library/Frameworks/OpenGL.framework/Libraries -lGL -lGLU -Xlinker -framework -Xlinker GLUT -Wno-deprecated

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// ---- BODY PROPERTIES ---- //
#define N 6
#define MASS 1.0 // 0.0000000000131946891 //estimate with density 1.05g per cm cube
#define DIAMETER_PS 1.0 // Diameter of polystyrene spheres 1 micron
#define DIAMETER_NIPAM 0.08 // Diameter of polyNIPAM microgel particles 80 nanometers
// globals to store positions, velocities, and forces
float p[N][3], v[N][3], f[N][3], mass[N];

// ---- DISPLAY PROPERTIES ---- //
// windows size
#define XWindowSize 1024
#define YWindowSize 1024
// OpenGL box size
float xMin = -4.0;
float xMax =  4.0;
float yMin = -4.0;
float yMax =  4.0;
float zMin = -4.0;
float zMax =  4.0;
// how many iterations you wait until you draw
#define DRAW 8000
// choose where you want your eye to be
#define EYEX 0.0
#define EYEY 0.0
#define EYEZ 8.2
#define FAR 50.0
#define RADIUS 6

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
#define NUMBER_OF_RUNS 2000
#define MAX_TOTAL_KINETIC_ENERGY 0.002
#define INITIAL_SEPARATION 1.1
#define NO 0
#define YES 1

// ---- RESULTS PROPERTIES ---- //
float octa_count = 0.0;
float tetra_count = 0.0;
float other_count = 0.0;

void set_initial_conditions()
{
	int i, j, ok_config;
	float mag, distance, dx, dy, dz;
	ok_config = NO;

	while(ok_config == NO)
	{
		for (i = 0; i < N; i++)
		{
			// initialize mass of bodies
			mass[i] = 1.0;
			// intitialize positions
			p[i][0] = ((float)rand()/(float)RAND_MAX) * (xMax - xMin) - ((xMax-xMin)/2);
			p[i][1] = ((float)rand()/(float)RAND_MAX) * (yMax - yMin) - ((yMax-yMin)/2);
			p[i][2] = ((float)rand()/(float)RAND_MAX) * (zMax - zMin) - ((yMax-yMin)/2);
			// initialize velocities
			mag = sqrt(p[i][0]*p[i][0]+p[i][1]*p[i][1]+p[i][2]*p[i][2]);
			v[i][0] = INITIAL_VELOCITY*(-p[i][0]/mag)*rand()/RAND_MAX;
			v[i][1] = INITIAL_VELOCITY*(-p[i][1]/mag)*rand()/RAND_MAX;
			v[i][2] = INITIAL_VELOCITY*(-p[i][2]/mag)*rand()/RAND_MAX;
		}
		// make sure each body is a minimum distance from all the others
		ok_config = YES;
		for(i = 0; i < N - 1; i++)
		{
			for(j = i + 1; j < N; j++)
			{
				dx = p[i][0]-p[j][0];
				dy = p[i][1]-p[j][1];
				dz = p[i][2]-p[j][2];
				distance = sqrt(dx*dx + dy*dy + dz*dz);
				if(distance <= INITIAL_SEPARATION) {
					// printf("bodies too close!\n");
					ok_config = NO;
				}
			}
		}
	}
}

void draw_box()
{
	glColor3d(.75, .75, .75);
	glLineWidth(5);
	// draw box
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMin, yMax, zMax);//left of window
	  glVertex3f(xMax, yMax, zMax);//bottom of window
	  glVertex3f(xMax, yMin, zMax);//right of window
	  glVertex3f(xMin, yMin, zMax);//top of window
	glEnd();//end drawing of line loop
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMin, yMax, zMin);//left of window
	  glVertex3f(xMax, yMax, zMin);//bottom of window
	  glVertex3f(xMax, yMin, zMin);//right of window
	  glVertex3f(xMin, yMin, zMin);//top of window
	glEnd();//end drawing of line loop
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMin, yMax, zMax);//left of window
	  glVertex3f(xMin, yMax, zMin);//bottom of window
	glEnd();//end drawing of line loop
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMin, yMin, zMax);//left of window
	  glVertex3f(xMin, yMin, zMin);//bottom of window
	glEnd();//end drawing of line loop
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMax, yMin, zMax);//left of window
	  glVertex3f(xMax, yMin, zMin);//bottom of window
	glEnd();//end drawing of line loop
	glBegin(GL_LINE_LOOP);//start drawing a line loop
	  glVertex3f(xMax, yMax, zMax);//left of window
	  glVertex3f(xMax, yMax, zMin);//bottom of window
	glEnd();//end drawing of line loop
}

void draw_picture()
{
	glClear(GL_COLOR_BUFFER_BIT);
	glClear(GL_DEPTH_BUFFER_BIT);
	glColor3d(0.0,1.0,0.0);

	int i;
	// draw bodies
	for (i = 0; i < N; i++)
	{
		glPushMatrix();
		glTranslatef(p[i][0], p[i][1], p[i][2]);
		// first entry is radius
		glutSolidSphere(0.15,20,20);
		glPopMatrix();
	}

	// // draw one sphere
	// glColor3d(0.0,0.0,1.0);
	// glPushMatrix();
	// glTranslatef(0.0, 0.0, 0.0);
	// // first entry is radius
	// glutSolidSphere(0.1, 20, 20);
	// glPopMatrix();
	draw_box();
	glutSwapBuffers();
}

void get_forces()
{
	float dx, dy, dz, squared_distance, distance;
	float force_mag;
	int i,j;

	// initialize forces to 0
	for (i = 0; i < N; i++)
	{
		f[i][0] = 0.0;
		f[i][1] = 0.0;
		f[i][2] = 0.0;
	}
    // loop through every body
	for (i = 0; i < N; i++)
	{
		// for each body, calculate distance and
		// force from every other body
		for (j = i+1; j < N; j++)
		{
			dx = p[j][0]-p[i][0];
			dy = p[j][1]-p[i][1];
			dz = p[j][2]-p[i][2];
			squared_distance = dx*dx + dy*dy + dz*dz;
			distance = sqrt(squared_distance);

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

}
void update_positions_and_velocities()
{
	int i;
	float dt;
	dt = DT;
	// update positions and velocities
	for(i = 0; i < N; i++)
	{
		v[i][0] += ((f[i][0]-DAMP*v[i][0])/mass[i])*dt;
		v[i][1] += ((f[i][1]-DAMP*v[i][1])/mass[i])*dt;
		v[i][2] += ((f[i][2]-DAMP*v[i][2])/mass[i])*dt;

		p[i][0] += v[i][0]*dt;
		p[i][1] += v[i][1]*dt;
		p[i][2] += v[i][2]*dt;
	}
}

float get_total_kinetic_energy()
{
	int i;
	float total_kinetic_energy;
	//calculate total kinetic energy
	total_kinetic_energy = 0.0;
	for(i = 0; i < N; i++)
	{
		// total kinetic energy = summation{i=1}{n} 1/2 m v_i^2
		total_kinetic_energy += 0.5*mass[i]*(v[i][0]*v[i][0] + v[i][1]*v[i][1] +v[i][2]*v[i][2]);
	}
	return total_kinetic_energy;
}

void identify_shape()
{
	int i,j;
	float dx, dy, dz, squared_distance, distance;

	// figure out which shape is formed
	float total_body_to_body_distance = 0.0;
	for(i = 0; i < N - 1; i++)
	{
		for(j = i + 1; j < N; j++)
		{
			dx = p[j][0]-p[i][0];
			dy = p[j][1]-p[i][1];
			dz = p[j][2]-p[i][2];
			squared_distance = dx*dx + dy*dy + dz*dz;
			distance = sqrt(squared_distance);
			total_body_to_body_distance += distance;
		}
	}
	// printf("total body distance: %.15f\n", total_body_to_body_distance);
	// theoretical distance: 16.2426
	if (total_body_to_body_distance < 16.5426 && 15.9426 < total_body_to_body_distance)
	{
		octa_count++;
	}
	// theoretical distance: 17.168
	else if (total_body_to_body_distance < 17.468 && 16.868 < total_body_to_body_distance)
	{
		tetra_count++;
	}
	else
	{
		other_count++;
	}

	// write object lengths to file for later plotting
	FILE *text_file = fopen("object-length.txt", "a");
	fprintf(text_file, "%.15f\n", total_body_to_body_distance);
	fclose(text_file);
}

void n_body()
{
	int run_count;
	float total_kinetic_energy;
	int draw_count = 0;
	int print_count = 0;
	float octa_rate, tetra_rate, other_rate;
	float num_experiments;

	for(run_count = 0; run_count < NUMBER_OF_RUNS; run_count++)
	{
		set_initial_conditions();
		total_kinetic_energy = 1.0;
		// stop updates when bodies have stopped moving
		while(total_kinetic_energy > MAX_TOTAL_KINETIC_ENERGY)
		{
			get_forces();
			update_positions_and_velocities();
			total_kinetic_energy = get_total_kinetic_energy();
			if(draw_count == DRAW)
			{
				draw_picture();
				draw_count = 0;
			}
			draw_count++;
			print_count++;
		}
		identify_shape();

		num_experiments = (float)run_count + 1.0;
		octa_rate = octa_count/num_experiments;
		tetra_rate = tetra_count/num_experiments;
		other_rate = other_count/num_experiments;
		printf("run count: %i\t octa_rate: %.2f\t tetra_rate: %.2f\t other_rate: %.2f\n",
				(run_count+1), octa_rate, tetra_rate, other_rate);
	}
}

void control()
{
	int draw_count = 0;
	set_initial_conditions();
	draw_picture();
	n_body();
	printf("\n DONE \n");
	while(1);
}

void Display(void)
{
	gluLookAt(EYEX, EYEY, EYEZ, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
	//        how close we are |
	//                           looking at   |
	//                                          turning your head
	glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	control();
}

void reshape(int w, int h)
{
	glViewport(0, 0, (GLsizei) w, (GLsizei) h);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glFrustum(-0.2, 0.2, -0.2, 0.2, 0.2, FAR);
	glMatrixMode(GL_MODELVIEW);
}

int main(int argc, char** argv)
{
	srand((unsigned int)time(NULL));
	glutInit(&argc,argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_DEPTH | GLUT_RGB);
	glutInitWindowSize(XWindowSize,YWindowSize);
	glutInitWindowPosition(0,0);
	glutCreateWindow("Self Assembly");
	GLfloat light_position[] = {1.0, 1.0, 1.0, 0.0};
	GLfloat light_ambient[]  = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[]  = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat lmodel_ambient[] = {0.2, 0.2, 0.2, 1.0};
	GLfloat mat_specular[]   = {1.0, 1.0, 1.0, 1.0};
	GLfloat mat_shininess[]  = {10.0};
	glClearColor(0.0, 0.0, 0.0, 0.0);
	glShadeModel(GL_SMOOTH);
	glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);
	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, lmodel_ambient);
	glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular);
	glMaterialfv(GL_FRONT, GL_SHININESS, mat_shininess);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	glutDisplayFunc(Display);
	glutReshapeFunc(reshape);
	glutMainLoop();
	return 0;
}
