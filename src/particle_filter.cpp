#include <random>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <iterator>

#include "particle_filter.h"

using namespace std;

#define   NUM_OF_PARTICLES    100 


void ParticleFilter::init(double x, double y, double theta, double std[])
 {

	// TODO: Set the number of particles. Initialize all particles to first position
	// (based on estimates of x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// ☆彡 lesson 14- 3: initialization 
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);


	std::default_random_engine gen;	///乱数ジェネレータ

	// resize the vectors of particles and weights
	num_particles = NUM_OF_PARTICLES;
	particles.resize(num_particles);

	// generate the particles
	for(auto& p: particles){
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1;
	}

	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) 
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	std::default_random_engine gen;	///乱数ジェネレータ

	//正規分布疑似乱数(engine)の準備
	std::normal_distribution<double> N_x(0, std_pos[0]);
	std::normal_distribution<double> N_y(0, std_pos[1]);
	std::normal_distribution<double> N_theta(0, std_pos[2]);


	for(auto& p: particles){

		//☆彡　lesson 14-8 : Calculate Prediction Step Quiz Explanation
		// [注]　θ'(dt) ≒　θ'*dt で 近似
		// 移動量に応じて内部座標を更新（予測）

		if( fabs(yaw_rate) < 0.0001){  // constant velocity
			  p.x += velocity * delta_t * cos(p.theta);
			  p.y += velocity * delta_t * sin(p.theta);

		} else{
			  p.x += velocity / yaw_rate * ( sin( p.theta + yaw_rate*delta_t ) - sin(p.theta) );
			  p.y += velocity / yaw_rate * ( cos( p.theta ) - cos( p.theta + yaw_rate*delta_t ) );
			  p.theta += yaw_rate * delta_t;
		}

		// predicted particles with added sensor noise
		p.x += N_x(gen);
		p.y += N_y(gen);
		p.theta += N_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) 
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for(auto& obs: observations){
		double minD = std::numeric_limits<float>::max();	///FLT_MAXで初期化する
	
		for(const auto& pred: predicted){
			double distance = dist(obs.x, obs.y, pred.x, pred.y);
			if( minD > distance){
					////Find closer values and update
					minD = distance;
					obs.id = pred.id;
					///predとobsの最短となるpred.idをobs.idで上書きする
					///※obsのidは初期化されない不定値である
			}
		}
	}
}


//手順：
// 1：各粒子のセンサー範囲内のランドマークを予測として収集
// 2：観測値を車両座標から地図座標に変換（回転と平行移動）http://planning.cs.uiuc.edu/node99.htmlから式3.3を参照
// 3：dataAssociation（予測、観測）を使用して、各観測のランドマーク指標を検索
// 4：多変量ガウス分布を用いて各粒子の重みを更新
// https://en.wikipedia.org/wiki/Multivariate_normal_distribution

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. 
	//   You can read more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. 
	//   Your particles are located according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 3.33
	//   http://planning.cs.uiuc.edu/node99.html


	//  [メモ]  main.cpp 宣言： pf.updateWeights(sensor_range, sigma_landmark, noisy_observations, map);
	//  double sensor_range = 50; // Sensor range [m]
    //  double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [σx [m], σy [m]]

	for(auto& p: particles){
		p.weight = 1.0;		//個々のパーティクルのweightは毎回ここで初期化する

		// step 1:　センサー有効範囲50m内にあるランドマークをピックアップ
		vector<LandmarkObs> predictions;
		for(const auto& lm: map_landmarks.landmark_list){
			double distance = dist(p.x, p.y, lm.x_f, lm.y_f);
			if( distance < sensor_range){ 
				predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
			}
		}

		// step 2: センサ座標系をMAP座標系に変換する（AssociationやWeight計算準備）
		vector<LandmarkObs> observations_map;
		double cos_theta = cos(p.theta);
		double sin_theta = sin(p.theta);

		for(const auto& obs: observations){
			//lesson 14-16: Landmarks Quiz Solution
			//パーティクル "p"（車の推定位置）

			//座標変換	| 地図X	|	|cosθ	-sinθ	|| ｾﾝｻX	|   |車両p.X|
 			//			|		| =	|				||		| + |　　   |
			//			| 地図Y	|	|sinθ	 cosθ	|| ｾﾝｻY	|   |車両p.Y|
			LandmarkObs mp;
			mp.x = obs.x * cos_theta - obs.y * sin_theta + p.x;
			mp.y = obs.x * sin_theta + obs.y * cos_theta + p.y;
			observations_map.push_back(mp);
			////観測点をマップ座標系に変換
		}

		// step 3: observations_mapにIDを割り付ける
		dataAssociation(predictions, observations_map);	

		// step 4: ウェイトの計算（MAP座標系で真値との差を利用）
		for(const auto& obs_m: observations_map){
		
			//☆彡  leseon 14-19 Particle Weights Solution

			//      obs_mのidに対応したmapの真値座標を取得して　obs_m との偏差を確認　（ずれの程度でWeightを調整）
			Map::single_landmark_s landmark = map_landmarks.landmark_list.at(obs_m.id-1);

			double x_term = pow(obs_m.x - landmark.x_f, 2) / (2 * pow(std_landmark[0], 2));
			double y_term = pow(obs_m.y - landmark.y_f, 2) / (2 * pow(std_landmark[1], 2));
			double exponent = (x_term + y_term); 
			double gauss_norm=1./(2 * M_PI * std_landmark[0] * std_landmark[1]);
			double  w = gauss_norm*exp(-exponent);
			p.weight *=  w;
		}

		weights.push_back(p.weight);	////リサンプル用にストックする

	 }

}


void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

//////////////////////////////////////////////////
//https://cpprefjp.github.io/reference/random/discrete_distribution.html
	std::random_device seed_gen;		///乱数ジェネレータ
	std::mt19937 engine(seed_gen());	///メルセンヌ・ツイスタ疑似乱数の準備

	//  確率列を定義: 浮動小数点数の範囲として定義する。合計値が1.0や10.0のような切りの良い数値である必要はない。
	//  大吉：1%  中吉：9%  小吉：25%  吉：25%  末吉：30%  凶：9%  大凶：1%
	//  std::vector<double> probabilities = {0.01 0.09 -.25 0.25 0.30 0.09 0.01 };
	//  ただし引数は確率ではなく重みで解釈（総和が1でなくともよい）
	std::vector<double>& probabilities=weights;

	// 分布オブジェクトを生成。
	// コンストラクタには、確率列のイテレータ範囲を指定する。
	std::discrete_distribution<> dist(probabilities.begin(),probabilities.end()	);
//////////////////////////////
	vector<Particle> resampled_particles;
	resampled_particles.resize(num_particles);

	// mt19937で割り付ける
	for(int i=0; i<num_particles; i++){
		int idx = dist(engine);
		resampled_particles[i] = particles[idx];
	}

	// 新旧入れ替え更新
	particles = resampled_particles;

	// pf内のメンバweightは　リサンプル後に解放する（個々のparticle内のweightはupdateWeightsでリセット）
	weights.clear();
}


/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////
Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations,
		                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length()-1);  // get rid of the trailing space
	return s;
}
