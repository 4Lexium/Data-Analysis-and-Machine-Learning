/// As input data is used a toy-MC sample consisting of four Gaussian-distributed
/// and linearly correlated input variables.
/// The methods to be used can be switched on and off by means of booleans, or
/// via the prompt command, for example:
///
///     root -l ./TMVAClassification.C\(\"Fisher,Likelihood\"\)
///
/// (note that the backslashes are mandatory)
/// If no method given, a default set of classifiers is used.
/// The output file "TMVA.root" can be analysed with the use of dedicated
/// macros (simply say: root -l <macro.C>), which can be conveniently
/// invoked through a GUI that will appear at the end of the run of this macro.
/// Launch the GUI via the command:
///
///     root -l ./TMVAGui.C
///
/// You can also compile and run the example with the following commands
///
///     make
///     ./TMVAClassification <Methods>

#include <cstdlib>
#include <iostream>
#include <string>

#include "TChain.h"
#include "TFile.h"
#include "TTree.h"
#include "TString.h"
#include "TObjString.h"
#include "TSystem.h"
#include "TROOT.h"
#include <TMath.h>

#include "TMVA/Factory.h"
#include "TMVA/DataLoader.h"
#include "TMVA/Tools.h"
#include "TMVA/TMVAGui.h"

using namespace std;

int TMVAClassification_D4H_DD(vector<TString> filepaths, TString TMVApath, TString datasetpath)
{
	// The explicit loading of the shared libTMVA is done in TMVAlogon.C, defined in .rootrc
	// if you use your private .rootrc, or run from a different directory, please copy the
	// corresponding lines from .rootrc

	// Methods to be processed can be given as an argument; use format:
	//
	//     mylinux~> root -l TMVAClassification.C\(\"myMethod1,myMethod2,myMethod3\"\)

	// Load library
	TMVA::Tools::Instance();

	cout << "\n==> Start BDT_TMVAClassification" << endl;

	//========================================//
	//  Load files into corresponding chains  //
	//========================================//
	TString treename = "DecayTree";
	TChain * sigchain = new TChain(treename);
	TChain * bkgchain = new TChain(treename);

	for (auto filepath : filepaths) {
		TString tempfilepath = filepath;
		if (tempfilepath.Contains("MC") || tempfilepath.Contains("mc")) {
			sigchain->Add(tempfilepath);
		}
		else if (tempfilepath.Contains("data")){
		 	bkgchain->Add(tempfilepath);
		}
	}
	int signentries = sigchain->GetEntries();
	int bkgnentries = bkgchain->GetEntries();
	cout << "Signal sample has " << signentries << " entries;\nBackground sample has "  << bkgnentries << " entries" << endl;


	//====================================================//
	//  Add TMVA classification discriminating variables  //
	//====================================================//
	TFile* outfile = TFile::Open( TMVApath, "RECREATE" );

	TMVA::Factory * factory = new TMVA::Factory( "TMVAClassification", outfile, "!V:!Silent:Color:DrawProgressBar:Transformations=I;D;P;G,D:AnalysisType=Classification" );

	TMVA::DataLoader * dataloader=new TMVA::DataLoader(datasetpath);

	// Define the input variables that shall be used for the MVA training
	// note that you may also use variable expressions, such as: "3*var1/var2*abs(var3)"
	// [all types of expressions that can also be parsed by TTree::Draw( "expression" )]
	//dataloader->AddVariable( "myvar1 := var1+var2", 'F' );
	//dataloader->AddVariable( "myvar2 := var1-var2", "Expression 2", "", 'F' );
	//dataloader->AddVariable( "var3",                "Variable 3", "units", 'F' );
	//dataloader->AddSpectator( "spec1 := var1*2",  "Spectator 1", "units", 'F' );


	dataloader->AddVariable( "log(B_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(B_ENDVERTEX_CHI2)"      , 'D');
	dataloader->AddVariable( "log(D0_ENDVERTEX_CHI2)"      , 'D');
	dataloader->AddVariable( "log(D0_FDCHI2_ORIVX)"      , 'D');
	dataloader->AddVariable( "log(D0_K_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi1_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi2_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi3_IPCHI2_OWNPV)"      , 'D');
	dataloader->AddVariable( "log(D0_pi1_PT)"      , 'D');
	dataloader->AddVariable( "log(D0_pi2_PT)"      , 'D');
	dataloader->AddVariable( "log(D0_pi3_PT)"      , 'D');
	dataloader->AddVariable( "log(p_PT)"      , 'D');
	dataloader->AddVariable("MaxLogPT := max(log(Lambda_p_PT), log(Lambda_pi_PT))", 'D');
	dataloader->AddVariable("MinLogPT := min(log(Lambda_p_PT), log(Lambda_pi_PT))", 'D');  // to reduce correlation


	// Assign samples weight
	Double_t sigweight = 1.0;
	Double_t bkgweight = 1.0;

	// You can add an arbitrary number of signal or background trees
	dataloader->AddSignalTree    ( sigchain, sigweight );
	dataloader->AddBackgroundTree( bkgchain, bkgweight );
	// dataloader->SetSignalWeightExpression("kinematic_weight*vertex_weight");


	// Tell the dataloader how to use the training and testing events
	//
	// If no numbers of events are given, half of the events in the tree are used
	// for training, and the other half for testing:
	//
	//dataloader->PrepareTrainingAndTestTree( mycuts, mycutb, "SplitMode=random:!V" );
	//
	// To also specify the number of testing events, use:
	//
	//    dataloader->PrepareTrainingAndTestTree( mycut,
	//         "NSigTrain=3000:NBkgTrain=3000:NSigTest=3000:NBkgTest=3000:SplitMode=Random:!V" );
	
	// TCut sigcut = "abs(B_M-5200)<100";
	// TCut bkgcut = "B_M>5800 && B_M<7000"; //Train on purpuse with the Higth SB of of B
	TCut sigcut = "1";
	TCut bkgcut = "B_M>5600 && B_M<7000";  // Train on purpuse with the Higth SB of of B 
	// dataloader->PrepareTrainingAndTestTree( sigcut, bkgcut, "SplitMode=random:!V" );
    dataloader->PrepareTrainingAndTestTree(sigcut, bkgcut,
                                           "nTrain_Signal=:nTrain_Background=:SplitMode=Random:NormMode=NumEvents:!V"); //MC splin into train and test equally 

	// ### Book MVA methods
	//
	// Please lookup the various method configuration options in the corresponding cxx files, eg:
	// src/MethoCuts.cxx, etc, or here: http://tmva.sourceforge.net/optionRef.html
	// it is possible to preset ranges in the option string in which the cut optimisation should be done:
	// "...:CutRangeMin[2]=-1:CutRangeMax[2]=1"...", where [2] is the third input variable

	// Use BDT

	factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
		"!H:!V:"
		"NTrees=600:"  
		"MaxDepth=3:"
		"MinNodeSize=2%:"         // against spiky behaviour
		"BoostType=AdaBoost:"
		"AdaBoostBeta=0.15:"        
		"UseBaggedBoost=True:"
		"BaggedSampleFraction=0.7:"
		"SeparationType=CrossEntropy:"  // GiniIndex is more sensitive to unequal distribution and sharp boundries
		"Shrinkage=0.1:"
		"nCuts=60:"   //better target variance
		"PruneMethod=NoPruning"
	);
	// factory->BookMethod(dataloader, TMVA::Types::kDNN, "BDT",
	// 	"!H:!V:"
	// 	"ErrorStrategy=CROSSENTROPY:"
	// 	"VarTransform=G:"
	// 	"Layout=RELU|256,RELU|128,RELU|64,Sigmoid:"   //RELU|16,Linear:"
	// 	"TrainingStrategy="
	// 	"LearningRate=1e-4,"
	// 	"BatchSize=64,"
	// 	"Momentum=0.9,"
	// 	"TestRepetitions=5,"
	// 	"MaxEpochs=200,"
	// 	"WeightDecay=1e-3,"
	// 	"DropoutFraction=0.3,"
	// 	"Optimizer=ADAM"
	// );


	// factory->BookMethod(dataloader, TMVA::Types::kDNN, "BDT",
    // "!H:!V:"
    // "ErrorStrategy=CROSSENTROPY:"
    // "VarTransform=N:"
    // "Layout=RELU|64,RELU|32,RELU|16,SIGMOID:"
    // "TrainingStrategy="
    //     "LearningRate=1e-3,"
    //     "ConvergenceSteps=20,"
    //     "BatchSize=64,"
    //     "TestRepetitions=2,"
    //     "MaxEpochs=300,"
    //     "WeightDecay=1e-5,"
    //     "Optimizer=ADAM"
	// );
    // factory->BookMethod(dataloader, TMVA::Types::kBDT, "BDT",
    //                     "!H:!V:NTrees=850:MinNodeSize=0.01%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning");
		// eacdh BDT tree should have max 3 levels to avoid overtraining, MaxDepth

		// "!H:!V:NTrees=850:MinNodeSize=2.5%:MaxDepth=3:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20" );

	// For an example of the category classifier usage, see: TMVAClassificationCategory
	//
	// --------------------------------------------------------------------------------------------------
	//  Now you can optimize the setting (configuration) of the MVAs using the set of training events
	// STILL EXPERIMENTAL and only implemented for BDT's !
	//
	//     factory->OptimizeAllMethods("SigEffAtBkg0.01","Scan");
	//     factory->OptimizeAllMethods("ROCIntegral","FitGA");
	//
	// --------------------------------------------------------------------------------------------------

	factory->TrainAllMethods();
	factory->TestAllMethods();
	factory->EvaluateAllMethods();

	outfile->Close();

	cout << "==> Wrote root file: " << outfile->GetName() << endl;
	cout << "==> BDT_TMVAClassification is done!" << endl;

	delete factory;
	delete dataloader;
	// Launch the GUI for the root macros
	if (!gROOT->IsBatch()) TMVA::TMVAGui( TMVApath );

	return 0;
}