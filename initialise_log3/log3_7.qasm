OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
u(pi/2,0,pi) q[7];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(pi/2,pi/4,-3*pi/4) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,pi/4,-pi) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(pi/2,0,3*pi/4) q[1];
u(0,2.191981133989078,-2.1919811339890782) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,-pi/4,3*pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(pi/2,pi/4,-pi) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,pi/4,-pi) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(pi/2,0,3*pi/4) q[1];
u(0,2.191981133989078,-2.1919811339890782) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,0,3*pi/4) q[3];
u(pi/2,0,pi) q[7];
u(pi/2,0,pi) q[8];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(pi/2,pi/4,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,pi/4,-pi) q[0];
u(0,2.191981133989078,-2.1919811339890782) q[2];
cx q[3],q[0];
u(0,0,-pi/4) q[0];
cx q[1],q[0];
u(0,0,pi/4) q[0];
cx q[3],q[0];
u(pi/2,0,3*pi/4) q[0];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,-pi/4,3*pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
u(pi/2,0,pi) q[7];
cx q[8],q[2];
u(pi/2,pi/4,-pi) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,pi/4,-pi) q[0];
u(0,2.191981133989078,-2.1919811339890782) q[2];
cx q[3],q[0];
u(0,0,-pi/4) q[0];
cx q[1],q[0];
u(0,0,pi/4) q[0];
cx q[3],q[0];
u(pi/2,0,3*pi/4) q[0];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,0,3*pi/4) q[2];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(pi/2,pi/4,-3*pi/4) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,pi/4,-pi) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(pi/2,0,3*pi/4) q[1];
u(0,2.191981133989078,-2.1919811339890782) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,-pi/4,3*pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(0,0,-pi/4) q[3];
cx q[6],q[3];
u(0,0,pi/4) q[3];
cx q[7],q[3];
u(pi/2,pi/4,-pi) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,pi/4,-pi) q[1];
cx q[2],q[1];
u(0,0,-pi/4) q[1];
cx q[0],q[1];
u(0,0,pi/4) q[1];
cx q[2],q[1];
u(pi/2,0,3*pi/4) q[1];
u(0,2.191981133989078,-2.1919811339890782) q[3];
cx q[1],q[3];
u(0,0,-pi/4) q[3];
cx q[4],q[3];
u(0,0,pi/4) q[3];
cx q[1],q[3];
u(pi/2,0,3*pi/4) q[3];
u(pi/2,0,pi) q[7];
u(0,1.4065829705916304,-1.4065829705916302) q[8];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(pi/2,pi/4,-3*pi/4) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,pi/4,-pi) q[0];
u(0,2.191981133989078,-2.1919811339890782) q[2];
cx q[3],q[0];
u(0,0,-pi/4) q[0];
cx q[1],q[0];
u(0,0,pi/4) q[0];
cx q[3],q[0];
u(pi/2,0,3*pi/4) q[0];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,-pi/4,3*pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
cx q[8],q[2];
u(0,0,-pi/4) q[2];
cx q[7],q[2];
u(0,0,pi/4) q[2];
cx q[8],q[2];
u(pi/2,pi/4,-pi) q[2];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,pi/4,-pi) q[0];
u(0,2.191981133989078,-2.1919811339890782) q[2];
cx q[3],q[0];
u(0,0,-pi/4) q[0];
cx q[1],q[0];
u(0,0,pi/4) q[0];
cx q[3],q[0];
u(pi/2,0,3*pi/4) q[0];
cx q[0],q[2];
u(0,0,-pi/4) q[2];
cx q[5],q[2];
u(0,0,pi/4) q[2];
cx q[0],q[2];
u(pi/2,0,3*pi/4) q[2];
u(pi/2,0,pi) q[8];