#include <stdio.h>
#include <cmath>
#include <vector>
#include <random>
#include <array>

#define _USE_MATH_DEFINES

float bxsize = 28.49;

float periodic_box(float p)
{
    float ret = std::fmod(std::abs(p), bxsize);
    if (p>=0)
        return ret;
    else
    {
        return bxsize - ret;
    }
}


int main(int argc, char** argv)
{

    long Npart=80000, Nclusters=2;
    int numdim=3, numit;
    int NX=4;
    float x,y,z,vx,vy,vz, r,phi,theta, a, deg=-0.4, mass=1000, G=4.3009172e-9;
    std::vector<std::array<float,3>> centers(Nclusters);
    FILE * outFile;
    outFile = fopen("rand_init.nyx", "w");
    
    float r_hi = bxsize/std::sqrt(2), r_lo = 0.01;
    float norm = (std::pow(r_hi, deg+1) - std::pow(r_lo, deg+1)) / (deg+1);
    float pmin = std::pow(r_lo, deg) / norm;
    float pmax = std::pow(r_hi, deg) / norm;
     
    //create random number generators
    std::random_device rd;
    std::mt19937 gen(rd());
    float K_avg = 0.5 * 4*M_PI*G/std::pow(norm,2)/(deg+1)/(deg+2)/(deg+4)*(std::pow(r_hi,deg+4) -std::pow(r_lo,deg+4));
    std::normal_distribution<> N_vel(K_avg*0.4, std::sqrt(2));
    std::uniform_real_distribution<> uniform(pmin, pmax); //Mpc

    for (auto & cntr:  centers){
        cntr[0] = periodic_box((uniform(gen)-pmin) / (pmax-pmin) * bxsize);
        cntr[1] = periodic_box((uniform(gen)-pmin) / (pmax-pmin) * bxsize);
        cntr[2] = periodic_box((uniform(gen)-pmin) / (pmax-pmin) * bxsize);
    }
    
    //write header
    std::fwrite(&Npart, sizeof(long), 1, outFile);
    std::fwrite(&numdim, sizeof(int), 1, outFile);
    std::fwrite(&NX, sizeof(int), 1, outFile);
    
    //geneate particles and write to file
    numit = int(Npart/Nclusters);
    for(auto &cntr : centers){ 
        for(int i=0; i<numit; i++){

            r = std::pow(uniform(gen)*norm, 1/deg);
            phi = (uniform(gen)-pmin) / (pmax-pmin) * 2*M_PI;
            theta = std::acos(2*(uniform(gen)-pmin)/(pmax-pmin) - 1);

            x = periodic_box(cntr[0] + r*std::sin(theta)*std::cos(phi));
            y = periodic_box(cntr[1] + r*std::sin(theta)*std::sin(phi));
            z = periodic_box(cntr[2] + r*std::cos(theta)); 
            vx = N_vel(gen);
            vy = N_vel(gen);
            vz = N_vel(gen);
           
           fwrite(&x, sizeof(float), 1, outFile);
           fwrite(&y, sizeof(float), 1, outFile);
           fwrite(&z, sizeof(float), 1, outFile);
           fwrite(&mass, sizeof(float), 1, outFile);
           fwrite(&vx, sizeof(float), 1, outFile);
           fwrite(&vy, sizeof(float), 1, outFile);
           fwrite(&vz, sizeof(float), 1, outFile);
        }
    }
    fclose(outFile);
}
