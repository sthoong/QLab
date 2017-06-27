
/Basic Neuro Network

/ Signmoid function for Threshold
sigmoid:{1%1+exp neg x}

/ Weight initialization
/ x: connection weight between x input and first neuron of next layer
/ y: weight leading to the second neuron and onward
wInit:{
  if[1=x;:"Number of input newrons must be greater than 1"];
  flip flip[r]-avg r:{[x;y]x?1.0}[y] each til x
 }

/ A sample for XOR problem
input:((0 0f);(0 1f);(1 0f);(1 1f));

/ add a Bias Neuron to each input
input:input,'1.0;

w:wInit[3;4];
v:wInit[5;1];
ffn:{[input;w;v]
  / Apply inputs and their weights to hidden layer
  z:sigmoid[input mmu w];
  
  / Use output from hidden layear to generate an output
  sigmoid[z mmu v]
}

ffn[input;w;v]


ffn:{[inputs;targets;lr;d]
  z:1.0,/:sigmoid[inputs mmu d`w];
  o:sigmoid[z mmu d`v];
  deltaO:(targets-o);
  deltaZ:1 /:$[deltaO;flip d`v]*z*1-z;
  `o`v`w!(o;d[`v]+lr*flip[z] mmu deltaO;d[`w]+lr*flip[inputs] mmu deltaZ)
 }
