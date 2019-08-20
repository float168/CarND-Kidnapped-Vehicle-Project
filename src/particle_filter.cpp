/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"


#define DEBUG

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 1000;  // Set the number of particles
  std::default_random_engine gen;

  constexpr double weight = 1.0;

  std::normal_distribution<double> distrib_x(x, std[0]);
  std::normal_distribution<double> distrib_y(y, std[1]);
  std::normal_distribution<double> distrib_theta(theta, std[2]);

  for (int i = 0; i < num_particles; ++i) {
    const double x_p = distrib_x(gen);
    const double y_p = distrib_y(gen);
    const double theta_p = distrib_theta(gen);
    particles.push_back(Particle{i, x_p, y_p, theta_p, weight});
    weights.push_back(weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */

  std::default_random_engine gen;
  std::normal_distribution<double> distrib_x(0.0, std_pos[0]);
  std::normal_distribution<double> distrib_y(0.0, std_pos[1]);
  std::normal_distribution<double> distrib_theta(0.0, std_pos[2]);

  for (auto& p : particles) {
    const double new_theta = p.theta + delta_t * yaw_rate;
    p.x += velocity / yaw_rate * (sin(new_theta) - sin(p.theta));
    p.y += velocity / yaw_rate * (cos(p.theta) - cos(new_theta));
    p.theta = new_theta;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   during the updateWeights phase.
   */
  if (predicted.empty()) {
    return;
  }

  for (auto& o : observations) {
    int nearest_id = -1;
    double nearest_d2 = std::numeric_limits<double>::max();

    for (const auto& p : predicted) {
      const double dx = p.x - o.x;
      const double dy = p.y - o.y;
      const double d2 = dx * dx + dy * dy;
      if (d2 < nearest_d2) {
        nearest_d2 = d2;
        nearest_id = p.id;
      }
    }

    o.id = nearest_id;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */

  vector<LandmarkObs> map_predicted;
  for (const auto l : map_landmarks.landmark_list) {
    map_predicted.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
  }

  for (auto& p : particles) {
    double weight = 1.0;

    vector<LandmarkObs> map_observations;
    for (const auto obs : observations) {
      const double x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
      const double y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
      map_observations.push_back(LandmarkObs{obs.id, x, y});
    }

    dataAssociation(map_predicted, map_observations);

#ifdef DEBUG
    std::cout << "observations: ";
    for (auto l : map_observations) {
      std::cout << "(" << l.id << ": " << l.x << " " << l.y << ") ";
    }
    std::cout << std::endl;
#endif

    for (const LandmarkObs obs : map_observations) {
      LandmarkObs assoc;
      for (const LandmarkObs pred : map_predicted) {
        if (obs.id == pred.id) {
          assoc = pred;
          break;
        }
      }

      const double prob = multivariate_prob(std_landmark[0], std_landmark[1],
                                            obs.x, obs.y,
                                            assoc.x, assoc.y);
      weight *= prob;
    }

    p.weight = weight;
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  for (unsigned int i = 0; i < particles.size(); ++i) {
    weights[i] = particles[i].weight;
  }

  std::default_random_engine engine;
  std::discrete_distribution<> sampler(weights.begin(), weights.end());

  vector<Particle> new_particles;
#ifdef DEBUG
  std::cout << "resample: ";
#endif
  for (unsigned int i = 0; i < particles.size(); ++i) {
    const int index = sampler(engine);
    new_particles.push_back(particles[index]);
#ifdef DEBUG
    std::cout << index << " ";
#endif
  }
#ifdef DEBUG
  std::cout << std::endl;
#endif

  particles = new_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
