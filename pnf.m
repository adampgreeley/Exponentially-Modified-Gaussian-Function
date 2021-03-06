function p=pnf(x) 
% cumulative probability for the normalized Gaussian
% x is scalar, vector or matrix
% f has the same shape as x
% version 1.0 2/10/99
% (c) Yves Lacouture, Universite Laval

a=find(x<0);
b=find(x>=0);
p=x;
m_sqrt2=sqrt(2);
p(b)=( (1+erf(x(b)./m_sqrt2)) ./ 2 );
p(a)=( (erfc(-x(a)./m_sqrt2)) ./ 2 );

