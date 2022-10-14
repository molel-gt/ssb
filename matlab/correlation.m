function [Fvv, Fss, Fsv] = correlation(data, T, boundary_condition, binsize)   
%By Zheng Ma, Department of Physics, Princeton University,
%zhengm@princeton.edu

%% Here data is an n*n*n scalar field, if what you have is a binary file, you need to first convert it into a scalar field and find the
%% corresponding threshold, you can try to run the following first: 

%% data = imgaussfilt3(YourData,1,'FilterSize',9); 
%% threshold=prctile(data(:), 100*(1-phi)); phi is the volume fraction of the solid phase, this keeps the original volume fraction unchanged 
%% any(data(:)==threshold);  This checks if the threshold is exactly the value of a certain pixel, which you can easily avoid.

% T is the threshold that determines the interfaces, the phase above the threshold is treated as the solid phase.  
% boundary_condition can be "YPeriod" or "NPeriod", depending on whether the configuration is uder periodic boundary conditions or not.
% binsize is optional, the default value is the length of one pixel.
% Examples: [Fvv Fss Fsv]=correlation(data,0.1,'YPeriod',2), [Fvv Fss Fsv]=correlation(data,0.1,'NPeriod')
% There are other parameters can be modified which are commented in the code.

% Please cite the paper "Precise algorithms to compute surface correlation functions of two-phase heterogeneous media
% and their applications" if you are publishing using this program. 

l = size(data,1);                    
P = data;
if nargin < 4
   binsize = 1.0;
end
m = int32(l/(2*binsize));      %number of bins, can be modified 
ln = 1000;      %number of random points along one sample line, can be modified 
Nsample = 1e5;  %number of samples used for computing S2, can be modified 
Delta = 100;    %threshold of 1/cos, can be modified 
Fsv = zeros(m,1);
Fss = zeros(m,1);
Fvv = zeros(m,1);
surface = zeros(l,1);         %position of intersections 
recicos = zeros(l,1);
g = zeros(4,1);         %compute gradient
BW = (sign(P-T)+1)./2;         %corresponding two-phase medium 
expectedboundary = {'YPeriod','NPeriod'};
    
boundary = validatestring(boundary_condition, expectedboundary);
switch boundary 
        case {'YPeriod'}
      
for i = 1:l
    for j = 1:l
        n = 0;
        for k = 1:l
            if((data(i,j,k)-T)*(data(i,j,mod(k,l)+1)-T)<0)
                n = n+1;
                g(1) = (P(mod(i+l,l)+1,j,k) - P(mod(i+l-2,l)+1,j,k))/2;
                g(2) = (P(i,mod(j+l,l)+1,k) - P(i,mod(j+l-2,l)+1,k))/2;
                g(3) = (P(i,j,mod(k+l,l)+1) - P(i,j,mod(k+l-2,l)+1))/2;
                g(4) = sqrt(sum(g(1:3).^2));
                surface(n) = k+abs(P(i,j,k)-T)/(abs(P(i,j,k)-T)+abs(P(i,j,mod(k,l)+1)-T));
                recicos(n) = g(4)/abs(g(3));
                if isnan(recicos(n)) || isinf(recicos(n))
                    recicos(n)=0;
                end
            end
        end
        
         for k1 = 1:n-1
             for k2 = (k1+1):n
                 d = abs(surface(k1)-surface(k2));
                 if d>0.5*l
                     d = l - d;
                 end
                     k = floor(d/binsize)+1;
                     Fss(k) = Fss(k) + 2*recicos(k1)*recicos(k2)*(recicos(k1)<Delta)*(recicos(k2)<Delta);
                
             end
          end
      
      for k1 = 1:ln
        
        px = rand*l+1;
        if ((P(i,j,int32(floor(px)))+(P(i,j,mod(int32(floor(px)),l)+1)-P(i,j,int32(floor(px))))*(px-floor(px)))<T)
            for k2 = 1:n
                d=abs(px-surface(k2));
                if d>0.5*l
                    d = l - d;
                end
                    k = floor(d/binsize)+1;
                    Fsv(k) = Fsv(k) + recicos(k2)*(recicos(k2)<Delta);
             end
        end
      end

    end
end
    for i = 1:m
      Fss(i) = Fss(i)/(l*l)/(2.0*binsize*l);
      Fsv(i) = Fsv(i)/(l*l)/(2.0*binsize*ln);
    end
    
  for i = 1:m
    	for j = 1:Nsample
            px = rand*l + 1;
            py = rand*l + 1;
            pz = rand*l + 1;
            theta1 = acos(rand);
            theta2 = rand*2*pi;
            nx = floor(px+(double(i)-0.5)*binsize*sin(theta1)*cos(theta2));
            ny = floor(py+(double(i)-0.5)*binsize*sin(theta1)*sin(theta2));
            nz = floor(pz+(double(i)-0.5)*binsize*cos(theta1));
            x0 = floor(px);
            y0 = floor(py);
            z0 = floor(pz);
            Fvv(i) = Fvv(i) + (1-BW(x0,y0,z0))*(1-BW(mod(nx+l-1,l)+1,mod(ny+l-1,l)+1,mod(nz+l-1,l)+1));
        end
        Fvv(i) = Fvv(i)/Nsample;
      
  end
        case {'NPeriod'}     
for i = 2:(l-1)
    for j = 2:(l-1)
        n = 0;
        for k = 2:(l-1)
            if((data(i,j,k)-T)*(data(i,j,k+1)-T)<0)
                n = n+1;
                g(1) = (P(mod(i+l,l)+1,j,k) - P(mod(i+l-2,l)+1,j,k))/2;
                g(2) = (P(i,mod(j+l,l)+1,k) - P(i,mod(j+l-2,l)+1,k))/2;
                g(3) = (P(i,j,mod(k+l,l)+1) - P(i,j,mod(k+l-2,l)+1))/2;
                g(4) = sqrt(sum(g(1:3).^2));
                surface(n) = k+abs(P(i,j,k)-T)/(abs(P(i,j,k)-T)+abs(P(i,j,k+1)-T));
                recicos(n) = g(4)/abs(g(3));
                if isnan(recicos(n)) || isinf(recicos(n))
                    recicos(n)=0;
                end
                
            end
        end
        
         for k1 = 1:(n-1)
             for k2 = (k1+1):n
                 d = abs(surface(k1)-surface(k2));
                 if d<0.5*l
                     k = floor(d/binsize)+1;
                     Fss(k) = Fss(k) + 2*recicos(k1)*recicos(k2)*(recicos(k1)<Delta)*(recicos(k2)<Delta);
                 end
                
             end
          end
      
      for k1 = 1:ln
        
        px = rand*l+1;
        if ((P(i,j,int32(floor(px)))+(P(i,j,mod(int32(floor(px)),l)+1)-P(i,j,int32(floor(px))))*(px-floor(px)))<T)
            for k2 = 1:n
                d=abs(px-surface(k2));
                if d<0.5*l
                    k = floor(d/binsize)+1;
                    Fsv(k) = Fsv(k) + recicos(k2)*(recicos(k2)<Delta);
                end
             end
        end
      end

    end
end
    for i = 1:m
      Fss(i) = Fss(i)/(l-1)^2/(2.0*binsize*(l-double(i)*binsize)); 
      Fsv(i) = Fsv(i)/(l-1)^2/(2.0*binsize*ln)*l/(l-double(i)*binsize);
    end
    
  for i = 1:m
    	for j = 1:Nsample
            px = rand*l + 1;
            py = rand*l + 1;
            pz = rand*l + 1;
            theta1 = acos(rand);
            theta2 = rand*2*pi;
            nx = floor(px+(double(i)-0.5)*binsize*sin(theta1)*cos(theta2));
            ny = floor(py+(double(i)-0.5)*binsize*sin(theta1)*sin(theta2));
            nz = floor(pz+(double(i)-0.5)*binsize*cos(theta1));
            x0 = floor(px);
            y0 = floor(py);
            z0 = floor(pz);
            if nx>=1 && nx<=l && ny>=1 && ny<=l && nz>=1 && nz<=l
               Fvv(i) = Fvv(i) + (1-BW(x0,y0,z0))*(1-BW(nx,ny,nz));
            end
        end
        Fvv(i) = Fvv(i)/Nsample/(1-1.5*double(i)*binsize/l+2/pi*(double(i)*binsize/l)^2-1/(4*pi)*(double(i)*binsize/l)^3);
  end
        otherwise
            error('Unknown boundary condition validation.')
end
    

  
Fvv(:,2) = Fvv; Fvv(:,1) = (linspace(1,double(m),double(m))-0.5)*binsize;
Fss(:,2) = Fss; Fss(:,1) = (linspace(1,double(m),double(m))-0.5)*binsize;
Fsv(:,2) = Fsv; Fsv(:,1) = (linspace(1,double(m),double(m))-0.5)*binsize;


    
end




    

    