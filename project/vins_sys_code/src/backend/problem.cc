#include <iostream>
#include <fstream>
#include <eigen3/Eigen/Dense>
#include <iomanip>
#include "backend/problem.h"
#include "utility/tic_toc.h"

#include <opencv2/imgproc/imgproc.hpp>
#include "parameters.h"

#ifdef USE_OPENMP

#include <omp.h>

#endif

using namespace std;

// define the format you want, you only need one instance of this...
const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

void writeToCSVfile(std::string name, Eigen::MatrixXd matrix) {
    std::ofstream f(name.c_str());
    f << matrix.format(CSVFormat);
}

namespace myslam {
namespace backend {
void Problem::LogoutVectorSize() {
    // LOG(INFO) <<
    //           "1 problem::LogoutVectorSize verticies_:" << verticies_.size() <<
    //           " edges:" << edges_.size();
}

Problem::Problem(ProblemType problemType) :
    problemType_(problemType) {
    LogoutVectorSize();
    verticies_marg_.clear();
}

Problem::~Problem() {
//    std::cout << "Problem IS Deleted"<<std::endl;
    global_vertex_id = 0;
}

bool Problem::AddVertex(std::shared_ptr<Vertex> vertex) {
    if (verticies_.find(vertex->Id()) != verticies_.end()) {
        // LOG(WARNING) << "Vertex " << vertex->Id() << " has been added before";
        return false;
    } else {
        verticies_.insert(pair<unsigned long, shared_ptr<Vertex>>(vertex->Id(), vertex));
    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        if (IsPoseVertex(vertex)) {
            ResizePoseHessiansWhenAddingPose(vertex);
        }
    }
    return true;
}

void Problem::AddOrderingSLAM(std::shared_ptr<myslam::backend::Vertex> v) {
    if (IsPoseVertex(v)) {
        v->SetOrderingId(ordering_poses_);
        idx_pose_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
        ordering_poses_ += v->LocalDimension();
    } else if (IsLandmarkVertex(v)) {
        v->SetOrderingId(ordering_landmarks_);
        ordering_landmarks_ += v->LocalDimension();
        idx_landmark_vertices_.insert(pair<ulong, std::shared_ptr<Vertex>>(v->Id(), v));
    }
}

void Problem::ResizePoseHessiansWhenAddingPose(shared_ptr<Vertex> v) {

    int size = H_prior_.rows() + v->LocalDimension();
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(v->LocalDimension()).setZero();
    H_prior_.rightCols(v->LocalDimension()).setZero();
    H_prior_.bottomRows(v->LocalDimension()).setZero();

}
void Problem::ExtendHessiansPriorSize(int dim)
{
    int size = H_prior_.rows() + dim;
    H_prior_.conservativeResize(size, size);
    b_prior_.conservativeResize(size);

    b_prior_.tail(dim).setZero();
    H_prior_.rightCols(dim).setZero();
    H_prior_.bottomRows(dim).setZero();
}

bool Problem::IsPoseVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPose") ||
            type == string("VertexSpeedBias");
}

bool Problem::IsLandmarkVertex(std::shared_ptr<myslam::backend::Vertex> v) {
    string type = v->TypeInfo();
    return type == string("VertexPointXYZ") ||
           type == string("VertexInverseDepth");
}

bool Problem::AddEdge(shared_ptr<Edge> edge) {
    if (edges_.find(edge->Id()) == edges_.end()) {
        edges_.insert(pair<ulong, std::shared_ptr<Edge>>(edge->Id(), edge));
    } else {
        // LOG(WARNING) << "Edge " << edge->Id() << " has been added before!";
        return false;
    }

    for (auto &vertex: edge->Verticies()) {
        vertexToEdge_.insert(pair<ulong, shared_ptr<Edge>>(vertex->Id(), edge));
    }
    return true;
}

vector<shared_ptr<Edge>> Problem::GetConnectedEdges(std::shared_ptr<Vertex> vertex) {
    vector<shared_ptr<Edge>> edges;
    auto range = vertexToEdge_.equal_range(vertex->Id());
    for (auto iter = range.first; iter != range.second; ++iter) {

        // 并且这个edge还需要存在，而不是已经被remove了
        if (edges_.find(iter->second->Id()) == edges_.end())
            continue;

        edges.emplace_back(iter->second);
    }
    return edges;
}

bool Problem::RemoveVertex(std::shared_ptr<Vertex> vertex) {
    //check if the vertex is in map_verticies_
    if (verticies_.find(vertex->Id()) == verticies_.end()) {
        // LOG(WARNING) << "The vertex " << vertex->Id() << " is not in the problem!" << endl;
        return false;
    }

    // 这里要 remove 该顶点对应的 edge.
    vector<shared_ptr<Edge>> remove_edges = GetConnectedEdges(vertex);
    for (size_t i = 0; i < remove_edges.size(); i++) {
        RemoveEdge(remove_edges[i]);
    }

    if (IsPoseVertex(vertex))
        idx_pose_vertices_.erase(vertex->Id());
    else
        idx_landmark_vertices_.erase(vertex->Id());

    vertex->SetOrderingId(-1);      // used to debug
    verticies_.erase(vertex->Id());
    vertexToEdge_.erase(vertex->Id());

    return true;
}

bool Problem::RemoveEdge(std::shared_ptr<Edge> edge) {
    //check if the edge is in map_edges_
    if (edges_.find(edge->Id()) == edges_.end()) {
        // LOG(WARNING) << "The edge " << edge->Id() << " is not in the problem!" << endl;
        return false;
    }

    edges_.erase(edge->Id());
    return true;
}

bool Problem::Solve(int iterations) {


    if (edges_.size() == 0 || verticies_.size() == 0) {
        std::cerr << "\nCannot solve problem without edges or verticies" << std::endl;
        return false;
    }

    TicToc t_solve;
    // 统计优化变量的维数，为构建 H 矩阵做准备
    SetOrdering();
    // 遍历edge, 构建 H 矩阵
    MakeHessian();
    // LM 初始化
    ComputeLambdaInit();
    // LM 算法迭代求解
    bool stop = false;
    int iter = 0;
    double last_chi_ = 1e20;
    //save the radius to txt
    std::ofstream radius_out;
    std::ofstream Lambda_out;
    std::ofstream chi_out;
    std::ofstream g_out;

    if(VERBOSE)
        if(RADIUS_CHI_G_OUTPUT)
        {
            if(solverType_ == SolverType::LM)
                Lambda_out.open("../bin/Lambda.txt",ios::trunc);
            else
                radius_out.open("../bin/Radius.txt",ios::trunc);
            chi_out.open("../bin/chi.txt",ios::trunc);
            g_out.open("../bin/g.txt",ios::trunc);
        }

    while (!stop) {
        if(solverType_ == SolverType::LM)
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Lambda= " << currentLambda_ <<std::endl;
        else
            std::cout << "iter: " << iter << " , chi= " << currentChi_ << " , Radius= " << current_region_raidus_ << std::endl;

        bool oneStepSuccess = false;
        int false_cnt = 0;
        while (!oneStepSuccess)  // 不断尝试 Lambda, 直到成功迭代一步
        {

            if ( false_cnt > 10)
            {
                if(VERBOSE)
                    if(STOP_REASON)
                        std::cout << "stop reason: false_cnt > 10" << std::endl;
                stop = true;
                break;
            }

            // setLambda
//            AddLambdatoHessianLM();
            // 第四步，解线性方程
            SolveLinearSystem();
            //
//            RemoveLambdaHessianLM();

            // 优化退出条件1： delta_x_ 很小则退出
//            if (delta_x_.squaredNorm() <= 1e-6 || false_cnt > 10)


            // 更新状态量
            UpdateStates();
            if(VERBOSE)
                if(RADIUS_CHI_G_OUTPUT)
                {
                    if(solverType_ == SolverType::LM)
                        Lambda_out<<currentLambda_<<std::endl;
                    else
                        radius_out<<current_region_raidus_<<std::endl;
                    chi_out<<currentChi_<<std::endl;
                    g_out<<b_backup.norm()<<std::endl;
                }
            // 判断当前步是否可行以及 LM 的 lambda 怎么更新, chi2 也计算一下
            oneStepSuccess = IsGoodStep();
            // 后续处理，
            if (oneStepSuccess) {
//                std::cout << " get one step success\n";

                std::cout<<"||delta_x||= " << delta_x_.norm()<<std::endl;
                // 在新线性化点 构建 hessian
                MakeHessian();
                // TODO:: 这个判断条件可以丢掉，条件 b_max <= 1e-12 很难达到，这里的阈值条件不应该用绝对值，而是相对值
//                double b_max = 0.0;
//                for (int i = 0; i < b_.size(); ++i) {
//                    b_max = max(fabs(b_(i)), b_max);
//                }
//                // 优化退出条件2： 如果残差 b_max 已经很小了，那就退出
//                stop = (b_max <= 1e-12);
                //currentChi_在IsGoodStep中已经更新
                if(last_chi_ - currentChi_ < 1e-5)
                {
                    if(VERBOSE)
                        if(STOP_REASON)
                            std::cout << "stop reason: last_chi_ - currentChi_ < 1e-5" << std::endl;
                    stop = true;
                }

                if(b_.lpNorm<Eigen::Infinity>() < stopThresholdGradient){
                    if(VERBOSE)
                        if(STOP_REASON)
                            std::cout << "stop reason: ||g||∞ < stopThresholdGradient" << std::endl;
                    stop = true;
                }
                false_cnt = 0;
            } else {
                false_cnt ++;
                RollbackStates();   // 误差没下降，回滚
            }
        }
//        std::cout<<"--------- false_cnt = "<< false_cnt<<std::endl;
        iter++;

        // 优化退出条件3： currentChi_ 跟第一次的 chi2 相比，下降了 1e6 倍则退出
//        if (sqrt(currentChi_) <= stopThresholdLM_)
//        if (sqrt(currentChi_) < 1e-15)
        if(iter >= iterations)
        {
            if(VERBOSE)
                if(STOP_REASON)
                    std::cout << "stop reason: iter >= iterations" << std::endl;
            stop = true;
        }
        last_chi_ = currentChi_;
    }
    if(VERBOSE)
        if(RADIUS_CHI_G_OUTPUT) {
            if (solverType_ == SolverType::LM)
            if (solverType_ == SolverType::LM)
                Lambda_out.close();
            else
                radius_out.close();
            chi_out.close();
            g_out.close();
        }
    std::cout << "problem solve cost: " << t_solve.toc() << " ms" << std::endl;
    std::cout << "   makeHessian cost: " << t_hessian_cost_ << " ms" << std::endl;
//    std::cout << "   reuse save cost: " << t_reuse_cost_ << " ms" << std::endl;

    T_HESSIAN_ALL += t_hessian_cost_;
    std::cout << "   makeHessian cost all: " << T_HESSIAN_ALL << " ms" << std::endl;

    t_hessian_cost_ = 0.;
    t_reuse_cost_ = 0 ;
    return true;
}

bool Problem::SolveGenericProblem(int iterations) {
    return true;
}

void Problem::SetOrdering() {

    // 每次重新计数
    ordering_poses_ = 0;
    ordering_generic_ = 0;
    ordering_landmarks_ = 0;

    // Note:: verticies_ 是 map 类型的, 顺序是按照 id 号排序的
    for (auto vertex: verticies_) {
        ordering_generic_ += vertex.second->LocalDimension();  // 所有的优化变量总维数

        if (problemType_ == ProblemType::SLAM_PROBLEM)    // 如果是 slam 问题，还要分别统计 pose 和 landmark 的维数，后面会对他们进行排序
        {
            AddOrderingSLAM(vertex.second);
        }

    }

    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        // 这里要把 landmark 的 ordering 加上 pose 的数量，就保持了 landmark 在后,而 pose 在前
        ulong all_pose_dimension = ordering_poses_;
        for (auto landmarkVertex : idx_landmark_vertices_) {
            landmarkVertex.second->SetOrderingId(
                landmarkVertex.second->OrderingId() + all_pose_dimension
            );
        }
    }

//    CHECK_EQ(CheckOrdering(), true);
}

bool Problem::CheckOrdering() {
    if (problemType_ == ProblemType::SLAM_PROBLEM) {
        int current_ordering = 0;
        for (auto v: idx_pose_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }

        for (auto v: idx_landmark_vertices_) {
            assert(v.second->OrderingId() == current_ordering);
            current_ordering += v.second->LocalDimension();
        }
    }
    return true;
}

void Problem::MakeHessian() {
    TicToc t_h;
    // 直接构造大的 H 矩阵
    ulong size = ordering_generic_;
    MatXX H(MatXX::Zero(size, size));
    VecX b(VecX::Zero(size));

    // TODO:: accelate, accelate, accelate
    std::vector<std::shared_ptr<Edge>> edges;
    for (auto edge : edges_) {
        edges.push_back(edge.second);
    }
////    std::cout<<"***** make hessian *******"<<std::endl;
#ifdef USE_OPENMP
#pragma omp parallel for
#endif
    for (int k = 0; k < edges.size(); ++k) {

        edges[k]->ComputeResidual();
        edges[k]->ComputeJacobians();

        // TODO:: robust cost
        auto jacobians = edges[k]->Jacobians();
        auto verticies = edges[k]->Verticies();
        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            if (v_i->IsFixed()) continue;    // Hessian 里不需要添加它的信息，也就是它的雅克比为 0

            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            // 鲁棒核函数会修改残差和信息矩阵，如果没有设置 robust cost function，就会返回原来的
            double drho;
            MatXX robustInfo(edges[k]->Information().rows(),edges[k]->Information().cols());
            edges[k]->RobustInfo(drho,robustInfo);

            MatXX JtW = jacobian_i.transpose() * robustInfo;
            edges[k]->jacobians_robust_.push_back(JtW);
            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];

                if (v_j->IsFixed()) continue;

                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                assert(v_j->OrderingId() != -1);
                MatXX hessian = JtW * jacobian_j;

                // 所有的信息矩阵叠加起来
#ifdef USE_OPENMP
#pragma omp critical
#endif
                H.block(index_i, index_j, dim_i, dim_j).noalias() += hessian;
                if (j != i) {
                    // 对称的下三角
#ifdef USE_OPENMP
#pragma omp critical
#endif
                    H.block(index_j, index_i, dim_j, dim_i).noalias() += hessian.transpose();

                }
            }
#ifdef USE_OPENMP
#pragma omp critical
#endif
            b.segment(index_i, dim_i).noalias() -= drho * jacobian_i.transpose()* edges[k]->Information() * edges[k]->Residual();
        }

    }
    Hessian_ = H;
    b_ = b;
    t_hessian_cost_ += t_h.toc();
    NUM_MAKE_HESSIAN ++;

    if(H_prior_.rows() > 0)
    {
//        std::cout<<"add the H_prior_ to H"<<std::endl;
        MatXX H_prior_tmp = H_prior_;
        VecX b_prior_tmp = b_prior_;

        /// 遍历所有 POSE 顶点，对fix的顶点设置相应的先验信息为 0 .  fix 外参数, SET PRIOR TO ZERO
        /// landmark 没有先验
        for (auto vertex: verticies_) {
            if (IsPoseVertex(vertex.second) && vertex.second->IsFixed() ) {
                int idx = vertex.second->OrderingId();
                int dim = vertex.second->LocalDimension();
                H_prior_tmp.block(idx,0, dim, H_prior_tmp.cols()).setZero();
                H_prior_tmp.block(0,idx, H_prior_tmp.rows(), dim).setZero();
                b_prior_tmp.segment(idx,dim).setZero();
//                std::cout << " fixed prior, set the Hprior and bprior part to zero, idx: "<<idx <<" dim: "<<dim<<std::endl;
            }
        }
        Hessian_.topLeftCorner(ordering_poses_, ordering_poses_) += H_prior_tmp;
        b_.head(ordering_poses_) += b_prior_tmp;
    }


    if(JACOBIAN_SCALING)
    {
        double eps = 1e-8;
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(Hessian_);
        Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
        Eigen::VectorXd S_sqrt = S.cwiseSqrt();
        MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
        Hessian_ = J.transpose() * J;
//        std::cout<<"the difference of Hessian"<<(Hessian_temp.inverse() * Hessian_).norm()<<std::endl;
//        std::cout<<"Hessian_:"<<std::endl<<Hessian_<<std::endl;
//        std::cout<<"JTJ"<<std::endl<<Hessian_temp<<std::endl;
        //backup the hessian and b before scale
        Hessian_backup = Hessian_;
        b_backup = b_;
//            std::cout<< "the difference norm of the hessian = " << (Hessian_.inverse() * Hessian_temp).norm()<<std::endl;
        VecX jacobian_scaling_vec = (J.array().square()).colwise().sum();
        jacobian_scaling_vec.array() = jacobian_scaling_vec.array().sqrt();
        jacobian_scaling_vec.array() += 1;
        jacobian_scaling_vec.array() = jacobian_scaling_vec.array().inverse();
        jacobian_scaling_ = jacobian_scaling_vec.asDiagonal();
//        std::cout<<"jacobi_scaling : \n"<<jacobian_scaling_.toDenseMatrix()<<std::endl;
//        std::cout<<"before scaling, Hessian : "<<std::endl<<Hessian_<<std::endl;
        Hessian_ = jacobian_scaling_ * Hessian_ * jacobian_scaling_;
//        std::cout<<"after scaling, Hessian: "<<std::endl<<Hessian_<<std::endl;
        //TODO 上面Hessian用J进行了重新计算，而b_没有重新计算,marginalize()这个函数也是
//        std::cout<<"before scaling, b_ : \n"<<b_<<std::endl;
        b_ = jacobian_scaling_ * b_;
//        std::cout<<"after scaling, b_ : \n"<<b_<<std::endl;

    } else{
//        std::cout<<"Hessian_:\n"<<Hessian_<<std::endl;
//        std::cout<<"b_ : \n"<<b_<<std::endl;
    }




    delta_x_ = VecX::Zero(size);  // initial delta_x = 0_n;


}

/*
 * Solve Hx = b, we can use PCG iterative method or use sparse Cholesky
 */
void Problem::SolveLinearSystem() {


    if (problemType_ == ProblemType::GENERIC_PROBLEM) {
        MatXX H = Hessian_;
//        std::cout<<"SolverLinearSystem, Hessian_ :\n"<<Hessian_<<std::endl;
        for (size_t i = 0; i < Hessian_.cols(); ++i) {
            H(i, i) += currentLambda_;
        }
        if(b_.transpose() * H * b_ <0)
            std::cerr<<"what a fuck, H is not semi-positive"<<std::endl;
        VecX hgn;
        hgn = H.ldlt().solve(b_);

        if(solverType_ == SolverType::LM )
        {
            // PCG solver
            // delta_x_ = PCGSolver(H, b_, H.rows() * 2);

            if(JACOBIAN_SCALING)
            {
                delta_x_ = jacobian_scaling_ * hgn;
                b_false = b_;//一旦当前步无效，b_不会重新计算，所以对缩放后的b_进行备份
                b_ = b_backup;//对b_进行解缩放
                Hessian_false = Hessian_;//Hessian_同理
                Hessian_ = Hessian_backup;
            }
            else{
                delta_x_ = hgn;

            }
//            std::cout<<"delta_x_ : \n"<<delta_x_<<std::endl;

        } else
        {
            //dog leg
            ComputeDoglegStep(hgn);
        }

    } else {

//        TicToc t_Hmminv;
        // step1: schur marginalization --> Hpp, bpp
        int reserve_size = ordering_poses_;
        int marg_size = ordering_landmarks_;
        MatXX Hmm = Hessian_.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = Hessian_.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = Hessian_.block(reserve_size, 0, marg_size, reserve_size);
        MatXX Hpp = Hessian_.block(0, 0, ordering_poses_, ordering_poses_);
        VecX bpp = b_.segment(0, reserve_size);
        VecX bmm = b_.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        MatXX DTD;
        if(OPTIMIZE_LM){
            if(DTD_SCALING)
            {
                VecX DTD_vec = Hessian_.colwise().sum();
                DTD = DTD_vec.asDiagonal();
                std::cout<<"DTD_vec:"<<DTD_vec.maxCoeff()<<std::endl;
                Hmm += currentLambda_ * DTD.block(reserve_size, reserve_size, marg_size, marg_size);
                Hpp += currentLambda_ * DTD.block(0, 0, ordering_poses_, ordering_poses_);
            }

        }
        // TODO:: use openMP
        for (auto landmarkVertex : idx_landmark_vertices_) {
            int idx = landmarkVertex.second->OrderingId() - reserve_size;
            int size = landmarkVertex.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }


        MatXX tempH = Hpm * Hmm_inv;

        H_pp_schur_ = Hpp - tempH * Hmp;

        b_pp_schur_ = bpp - tempH * bmm;

        // step2: solve Hpp * delta_x = bpp
        VecX delta_x_pp(VecX::Zero(reserve_size));
        //TODO 这里相当于令D=I,可以改成JTJ的对角线元素，可以参考ceres的代码
//        if(solverType_ == SolverType::LM)
//        {
//            for (ulong i = 0; i < ordering_poses_; ++i){
//                H_pp_schur_(i, i) += currentLambda_; // LM Method
//            }
//        }

//        ceres 在dogleg算法里计算gauss newton step时也用了LM的方法


        for (ulong i = 0; i < ordering_poses_; ++i){
            H_pp_schur_(i, i) += currentLambda_; // LM Method
        }

//        if(OPTIMIZE_LM){
//            VecX DTD_vec = Hessian_.colwise().sum();
//            DTD = DTD_vec.asDiagonal();
//            H_pp_schur_ += currentLambda_ * DTD.block(0, 0, ordering_poses_, ordering_poses_);
//        }


        // TicToc t_linearsolver;
        VecX hgn(VecX::Zero(delta_x_.size()));

        delta_x_pp =  H_pp_schur_.ldlt().solve(b_pp_schur_);//  SVec.asDiagonal() * svd.matrixV() * Ub;    
        hgn.head(reserve_size) = delta_x_pp;
        // std::cout << " Linear Solver Time Cost: " << t_linearsolver.toc() << std::endl;

        // step3: solve Hmm * delta_x = bmm - Hmp * delta_x_pp;
        VecX delta_x_ll(marg_size);
        delta_x_ll = Hmm_inv * (bmm - Hmp * delta_x_pp);
        hgn.tail(marg_size) = delta_x_ll;


        if(solverType_ == SolverType::LM)
        {

            if(JACOBIAN_SCALING)
            {
                delta_x_ = jacobian_scaling_ * hgn;
                b_false = b_;//一旦当前步无效，b_不会重新计算，所以对缩放后的b_进行备份
                b_ = b_backup;//对b_进行解缩放
                Hessian_false = Hessian_;
                Hessian_ = Hessian_backup;

            }
            else
                delta_x_ = hgn;
        }
        else
            ComputeDoglegStep(hgn);



//        std::cout << "schur time cost: "<< t_Hmminv.toc()<<std::endl;
    }

}
void Problem::ComputeDoglegStep(const VecX hgn){


    //2. compute hdl
    scale_ = 0; //使用之前先置０
    if(hgn.norm() <= current_region_raidus_ )
    {
        if(VERBOSE)
            if(HDL_CHOOSE)
                std::cout<<"hdl choose: hgn.norm() <= current_region_raidus_, ||hgn||="<<hgn.norm()<<",region_radius="<<current_region_raidus_<<std::endl;
        delta_x_ = hgn;
        scale_ = currentChi_;
    }
    else{
        // compute gradient decent step
        if(!reuse_ || hsd_.size()<1)//加上第二个判断条件是因为第一次如果是case1然后又失效的话，hsd此时无值，reuse_又等于true
        {
            hsd_ = b_; //F's gradient
            alpha_ = hsd_.squaredNorm() / (hsd_.transpose() * Hessian_ * hsd_);
            //TODO alpha都是0,为什么,检查计算过程中的数值问题
            std::cout<<"alpha = "<<alpha_<<std::endl;
            a_ = alpha_ * hsd_;
        }


        if(a_.norm() >= current_region_raidus_)
        {
            if(VERBOSE)
                if(HDL_CHOOSE)
                    std::cout<<"hdl choose: a_.norm() >= current_region_raidus_ "<<std::endl;
            //TODO check the hsd_.norm
            delta_x_ = current_region_raidus_  * hsd_.normalized();
            scale_ = current_region_raidus_ * (2 * a_.norm() - current_region_raidus_) / (2 * alpha_);
        }
        else{
            if(VERBOSE)
                if(HDL_CHOOSE)
                    std::cout<<"hdl choose: hdl = alpha_ * hsd_ + beta *(hgn - alpha_ * hsd_) "<<std::endl;
            VecX b = hgn;
            double c = a_.transpose() * b - a_.squaredNorm();
            double beta;
            double d = sqrt(c*c + (b-a_).squaredNorm() * (pow(current_region_raidus_,2) - a_.squaredNorm()));
            beta = (c <= 0) ? (d - c) / (b-a_).squaredNorm()
                            : (pow(current_region_raidus_,2) - a_.squaredNorm()) / (c + d);

            delta_x_ = alpha_ * hsd_ + beta * (hgn - alpha_ * hsd_);
            assert(abs(delta_x_.norm() - current_region_raidus_) < 1e-5);
            scale_ = 0.5 * alpha_ *pow((1 - beta),2) * hsd_.squaredNorm() + beta * (2 - beta) * currentChi_;
        }
    }
    if(scale_ < 0)
        std::cerr<<"what a fuck, no decent"<<std::endl;

}


void Problem::UpdateStates() {

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->BackUpParameters();    // 保存上次的估计值

        ulong idx = vertex.second->OrderingId();
        ulong dim = vertex.second->LocalDimension();
        VecX delta = delta_x_.segment(idx, dim);
        vertex.second->Plus(delta);
    }

    // update prior
    if (err_prior_.rows() > 0) {
        // BACK UP b_prior_
        b_prior_backup_ = b_prior_;
        err_prior_backup_ = err_prior_;

        /// update with first order Taylor, b' = b + \frac{\delta b}{\delta x} * \delta x
        /// \delta x = Computes the linearized deviation from the references (linearization points)
        b_prior_ -= H_prior_ * delta_x_.head(ordering_poses_);       // update the error_prior
        err_prior_ = -Jt_prior_inv_ * b_prior_.head(ordering_poses_ - 15);

//        std::cout << "                : "<< b_prior_.norm()<<" " <<err_prior_.norm()<< std::endl;
//        std::cout << "     delta_x_ ex: "<< delta_x_.head(6).norm() << std::endl;
    }

}

void Problem::RollbackStates() {

    if(JACOBIAN_SCALING)
    {
        b_ = b_false;
        Hessian_ = Hessian_false;
//        std::cout<<"rollback b_: \n"<<b_<<std::endl;
//        std::cout<<"rollback Hessian_ \n"<<Hessian_<<std::endl;
    }

    // update vertex
    for (auto vertex: verticies_) {
        vertex.second->RollBackParameters();
    }

    // Roll back prior_
    if (err_prior_.rows() > 0) {
        b_prior_ = b_prior_backup_;
        err_prior_ = err_prior_backup_;
    }
}

/// LM or DOGLEG
void Problem::ComputeLambdaInit() {

    currentLambda_ = -1.;
    currentChi_ = 0.0;

    for (auto edge: edges_) {
        currentChi_ += edge.second->RobustChi2();
    }
    if (err_prior_.rows() > 0)
        currentChi_ += err_prior_.squaredNorm();
    currentChi_ *= 0.5;

//    stopThresholdLM_ = 1e-10 * currentChi_;          // 迭代条件为 误差下降 1e-6 倍

    double maxDiagonal = 0;
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        maxDiagonal = std::max(fabs(Hessian_(i, i)), maxDiagonal);
    }

    maxDiagonal = std::min(5e10, maxDiagonal);
    double tau;
    if(JACOBIAN_SCALING)
        tau = 1e-5;
    else
        tau = 1e-5;
    currentLambda_ = tau * maxDiagonal;
//    if(OPTIMIZE_LM)
//    {
//        currentLambda_ = 1e-4;
//    }
//    if(solverType_ == SolverType::DOG_LEG)
//    {
//        currentLambda_ = 1e-8;
//    }
    min_Lambda_ = 1e-8;
    //TODO dogleg 的current_region_raidus_的初始值参考下ceres代码
//    current_region_raidus_ = 1 / currentLambda_;
    current_region_raidus_ = 1e2;
    reuse_ = false;

    std::cout << "maxDiagonal = "<<maxDiagonal<<"   "<<"currentLambda_ = "<<currentLambda_<<std::endl;
}

void Problem::AddLambdatoHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) += currentLambda_;
    }
}

void Problem::RemoveLambdaHessianLM() {
    ulong size = Hessian_.cols();
    assert(Hessian_.rows() == Hessian_.cols() && "Hessian is not square");
    // TODO:: 这里不应该减去一个，数值的反复加减容易造成数值精度出问题？而应该保存叠加lambda前的值，在这里直接赋值
    for (ulong i = 0; i < size; ++i) {
        Hessian_(i, i) -= currentLambda_;
    }
}
//LM or DOGLEG
bool Problem::IsGoodStep() {


    // recompute residuals after update state
    double tempChi = 0.0;
    for (auto edge: edges_) {
        edge.second->ComputeResidual();
        tempChi += edge.second->RobustChi2();
    }
    if (err_prior_.size() > 0)
        tempChi += err_prior_.squaredNorm();
    tempChi *= 0.5;          // 1/2 * err^2

    if(solverType_ == SolverType::LM)
    {
        scale_ = 0;//使用之前置0
        //    scale = 0.5 * delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
//    scale += 1e-3;    // make sure it's non-zero :)
//        std::cout<<"***********compute the scale"<<std::endl;
        if(JACOBIAN_SCALING)//currentLambda_是在J被scale后计算的，到这里时b_和delta_x_都已经解缩放了，因此不能和currentLambda一起计算，所以不能用第二个公式
            scale_ = -delta_x_.transpose() * (-b_ + 0.5 * Hessian_ * delta_x_);
        else
            scale_ = 0.5* delta_x_.transpose() * (currentLambda_ * delta_x_ + b_);
        scale_ += 1e-6;    // make sure it's non-zero :)
//        std::cout<<"***********compute the scale done"<<std::endl;

    } else{
//        scale_ = 0;
//        auto scale_temp = b_.transpose() * delta_x_ - 0.5 * delta_x_.transpose() * Hessian_ * delta_x_;
//        std::cout<<"scale_ = "<<scale_<<"       scale_temp = "<<scale_temp<<std::endl;
//        scale_ = scale_temp(0);
        scale_ += 1e-6;
    }//dog leg在ComputeDoglegStep()函数中已经计算过scale_

    double rho = (currentChi_ - tempChi) / scale_;

    if(solverType_ == SolverType::LM)
    {
        if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
        {
            double alpha = 1. - pow((2 * rho - 1), 3);
            alpha = std::min(alpha, 2. / 3.);
            double scaleFactor = (std::max)(1. / 3., alpha);
            currentLambda_ *= scaleFactor;
            ni_ = 2;
            currentChi_ = tempChi;
            return true;
        } else {
            currentLambda_ *= ni_;
            ni_ *= 2;
            return false;
        }
    } else{
//        if (rho > 0 && isfinite(tempChi))   // last step was good, 误差在下降
//        {
//            double alpha_ = 1. - pow((2 * rho - 1), 3);
//            alpha_ = std::min(alpha_, 2. / 3.);
//            double scaleFactor = (std::max)(1. / 3., alpha_);
//            currentLambda_ *= scaleFactor;
//            ni_ = 2;
//            currentChi_ = tempChi;
//            current_region_raidus_ = 1 / currentLambda_;
//            return true;
//        } else {
//            currentLambda_ *= ni_;
//            ni_ *= 2;
//            current_region_raidus_ = 1 / currentLambda_;
//            return false;
//        }


        if(rho < 0 || !isfinite(tempChi))
        {
            //LM Lambda update method
            currentLambda_ *= ni_;
            ni_ *= 2;
//           dogleg Lambda update method
//            currentLambda_ *= ni_;

            current_region_raidus_ *= 0.5;
            reuse_ = true;
            return false;
        }

        else
        {
            if(rho < 0.25)
            {
                current_region_raidus_ *= 0.5;
            }
            if(rho > 0.75)
            {
                current_region_raidus_ = max(current_region_raidus_, 3 * delta_x_.norm());
            }
            //LM Lambda update method
            double alpha = 1. - pow((2 * rho - 1), 3);
            alpha = std::min(alpha, 2. / 3.);
            double scaleFactor = (std::max)(1. / 3., alpha);
            currentLambda_ *= scaleFactor;
//            dog leg lambda update method
//            currentLambda_ = std::max(min_Lambda_, 2.0 * currentLambda_ / ni_);
            currentChi_ = tempChi;
            reuse_ = false;
            return true;
        }



    }

}

/** @brief conjugate gradient with perconditioning
 *
 *  the jacobi PCG method
 *
 */
VecX Problem::PCGSolver(const MatXX &A, const VecX &b, int maxIter = -1) {
    assert(A.rows() == A.cols() && "PCG solver ERROR: A is not a_ square matrix");
    int rows = b.rows();
    int n = maxIter < 0 ? rows : maxIter;
    VecX x(VecX::Zero(rows));
    MatXX M_inv = A.diagonal().asDiagonal().inverse();
    VecX r0(b);  // initial r = b - A*0 = b
    VecX z0 = M_inv * r0;
    VecX p(z0);
    VecX w = A * p;
    double r0z0 = r0.dot(z0);
    double alpha = r0z0 / p.dot(w);
    VecX r1 = r0 - alpha * w;
    int i = 0;
    double threshold = 1e-6 * r0.norm();
    while (r1.norm() > threshold && i < n) {
        i++;
        VecX z1 = M_inv * r1;
        double r1z1 = r1.dot(z1);
        double belta = r1z1 / r0z0;
        z0 = z1;
        r0z0 = r1z1;
        r0 = r1;
        p = belta * p + z1;
        w = A * p;
        alpha = r1z1 / p.dot(w);
        x += alpha * p;
        r1 -= alpha * w;
    }
    return x;
}

/*
 *  marg 所有和 frame 相连的 edge: imu factor, projection factor
 *  如果某个landmark和该frame相连，但是又不想加入marg, 那就把改edge先去掉
 *
 */
bool Problem::Marginalize(const std::vector<std::shared_ptr<Vertex> > margVertexs, int pose_dim) {

    SetOrdering();
    /// 找到需要 marg 的 edge, margVertexs[0] is frame, its edge contained pre-intergration
    std::vector<shared_ptr<Edge>> marg_edges = GetConnectedEdges(margVertexs[0]);

    std::unordered_map<int, shared_ptr<Vertex>> margLandmark;
    // 构建 Hessian 的时候 pose 的顺序不变，landmark的顺序要重新设定
    int marg_landmark_size = 0;
//    std::cout << "\n marg edge 1st id: "<< marg_edges.front()->Id() << " end id: "<<marg_edges.back()->Id()<<std::endl;
    for (size_t i = 0; i < marg_edges.size(); ++i) {
//        std::cout << "marg edge id: "<< marg_edges[i]->Id() <<std::endl;
        auto verticies = marg_edges[i]->Verticies();
        for (auto iter : verticies) {
            if (IsLandmarkVertex(iter) && margLandmark.find(iter->Id()) == margLandmark.end()) {
                iter->SetOrderingId(pose_dim + marg_landmark_size);
                margLandmark.insert(make_pair(iter->Id(), iter));
                marg_landmark_size += iter->LocalDimension();
            }
        }
    }
//    std::cout << "pose dim: " << pose_dim <<std::endl;
    //这里只对与marg掉的变量有关的边求hessian，其他无关的边在后续优化的时候会添加
    int cols = pose_dim + marg_landmark_size;
    /// 构建误差 H 矩阵 H = H_marg + H_pp_prior
    MatXX H_marg(MatXX::Zero(cols, cols));
    VecX b_marg(VecX::Zero(cols));
    int ii = 0;
    for (auto edge: marg_edges) {
        edge->ComputeResidual();
        edge->ComputeJacobians();
        auto jacobians = edge->Jacobians();
        auto verticies = edge->Verticies();
        ii++;

        assert(jacobians.size() == verticies.size());
        for (size_t i = 0; i < verticies.size(); ++i) {
            auto v_i = verticies[i];
            auto jacobian_i = jacobians[i];
            ulong index_i = v_i->OrderingId();
            ulong dim_i = v_i->LocalDimension();

            double drho;
            MatXX robustInfo(edge->Information().rows(),edge->Information().cols());
            edge->RobustInfo(drho,robustInfo);

            for (size_t j = i; j < verticies.size(); ++j) {
                auto v_j = verticies[j];
                auto jacobian_j = jacobians[j];
                ulong index_j = v_j->OrderingId();
                ulong dim_j = v_j->LocalDimension();

                MatXX hessian = jacobian_i.transpose() * robustInfo * jacobian_j;

                assert(hessian.rows() == v_i->LocalDimension() && hessian.cols() == v_j->LocalDimension());
                // 所有的信息矩阵叠加起来
                H_marg.block(index_i, index_j, dim_i, dim_j) += hessian;
                if (j != i) {
                    // 对称的下三角
                    H_marg.block(index_j, index_i, dim_j, dim_i) += hessian.transpose();
                }
            }
            b_marg.segment(index_i, dim_i) -= drho * jacobian_i.transpose() * edge->Information() * edge->Residual();
        }

    }
        std::cout << "edge factor cnt: " << ii <<std::endl;



    /// marg landmark
    int reserve_size = pose_dim;
    if (marg_landmark_size > 0) {
        int marg_size = marg_landmark_size;
        MatXX Hmm = H_marg.block(reserve_size, reserve_size, marg_size, marg_size);
        MatXX Hpm = H_marg.block(0, reserve_size, reserve_size, marg_size);
        MatXX Hmp = H_marg.block(reserve_size, 0, marg_size, reserve_size);
        VecX bpp = b_marg.segment(0, reserve_size);
        VecX bmm = b_marg.segment(reserve_size, marg_size);

        // Hmm 是对角线矩阵，它的求逆可以直接为对角线块分别求逆，如果是逆深度，对角线块为1维的，则直接为对角线的倒数，这里可以加速
        MatXX Hmm_inv(MatXX::Zero(marg_size, marg_size));
        // TODO:: use openMP
        for (auto iter: margLandmark) {
            int idx = iter.second->OrderingId() - reserve_size;
            int size = iter.second->LocalDimension();
            Hmm_inv.block(idx, idx, size, size) = Hmm.block(idx, idx, size, size).inverse();
        }

        MatXX tempH = Hpm * Hmm_inv;
        MatXX Hpp = H_marg.block(0, 0, reserve_size, reserve_size) - tempH * Hmp;
        bpp = bpp - tempH * bmm;
        H_marg = Hpp;
        b_marg = bpp;
    }

    VecX b_prior_before = b_prior_;
    if(H_prior_.rows() > 0)
    {
        H_marg += H_prior_;
        b_marg += b_prior_;
    }

    /// marg frame and speedbias
    int marg_dim = 0;

    // index 大的先移动
    for (int k = margVertexs.size() -1 ; k >= 0; --k)
    {
        int idx = margVertexs[k]->OrderingId();
        int dim = margVertexs[k]->LocalDimension();
//        std::cout << k << " "<<idx << std::endl;
        marg_dim += dim;
        // move the marg pose to the Hmm bottown right
        // 将 row i 移动矩阵最下面
        Eigen::MatrixXd temp_rows = H_marg.block(idx, 0, dim, reserve_size);
        Eigen::MatrixXd temp_botRows = H_marg.block(idx + dim, 0, reserve_size - idx - dim, reserve_size);
        H_marg.block(idx, 0, reserve_size - idx - dim, reserve_size) = temp_botRows;
        H_marg.block(reserve_size - dim, 0, dim, reserve_size) = temp_rows;

        // 将 col i 移动矩阵最右边
        Eigen::MatrixXd temp_cols = H_marg.block(0, idx, reserve_size, dim);
        Eigen::MatrixXd temp_rightCols = H_marg.block(0, idx + dim, reserve_size, reserve_size - idx - dim);
        H_marg.block(0, idx, reserve_size, reserve_size - idx - dim) = temp_rightCols;
        H_marg.block(0, reserve_size - dim, reserve_size, dim) = temp_cols;

        Eigen::VectorXd temp_b = b_marg.segment(idx, dim);
        Eigen::VectorXd temp_btail = b_marg.segment(idx + dim, reserve_size - idx - dim);
        b_marg.segment(idx, reserve_size - idx - dim) = temp_btail;
        b_marg.segment(reserve_size - dim, dim) = temp_b;
    }

    double eps = 1e-8;
    int m2 = marg_dim;
    int n2 = reserve_size - marg_dim;   // marg pose
    Eigen::MatrixXd Amm = 0.5 * (H_marg.block(n2, n2, m2, m2) + H_marg.block(n2, n2, m2, m2).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm);
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * Eigen::VectorXd(
            (saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0)).asDiagonal() *
                              saes.eigenvectors().transpose();

    Eigen::VectorXd bmm2 = b_marg.segment(n2, m2);
    Eigen::MatrixXd Arm = H_marg.block(0, n2, n2, m2);
    Eigen::MatrixXd Amr = H_marg.block(n2, 0, m2, n2);
    Eigen::MatrixXd Arr = H_marg.block(0, 0, n2, n2);
    Eigen::VectorXd brr = b_marg.segment(0, n2);
    Eigen::MatrixXd tempB = Arm * Amm_inv;
    H_prior_ = Arr - tempB * Amr;
    b_prior_ = brr - tempB * bmm2;

    //cholesky分解，从hessian矩阵分解出雅可比，通过雅可比可以从b_prior中恢复出error_prior
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(H_prior_);
    Eigen::VectorXd S = Eigen::VectorXd((saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0));
    Eigen::VectorXd S_inv = Eigen::VectorXd(
            (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0));

    Eigen::VectorXd S_sqrt = S.cwiseSqrt();
    Eigen::VectorXd S_inv_sqrt = S_inv.cwiseSqrt();
    Jt_prior_inv_ = S_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    err_prior_ = -Jt_prior_inv_ * b_prior_;

    MatXX J = S_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    H_prior_ = J.transpose() * J;
    MatXX tmp_h = MatXX( (H_prior_.array().abs() > 1e-9).select(H_prior_.array(),0) );
    H_prior_ = tmp_h;

    // std::cout << "my marg b prior: " <<b_prior_.rows()<<" norm: "<< b_prior_.norm() << std::endl;
    // std::cout << "    error prior: " <<err_prior_.norm() << std::endl;

    // remove vertex and remove edge
    for (size_t k = 0; k < margVertexs.size(); ++k) {
        RemoveVertex(margVertexs[k]);
    }

    for (auto landmarkVertex: margLandmark) {
        RemoveVertex(landmarkVertex.second);
    }

    return true;

}

}
}





