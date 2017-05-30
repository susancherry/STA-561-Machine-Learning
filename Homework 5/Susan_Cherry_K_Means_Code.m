function [new, k_means] = K_Means_Code(image,num_k)
%K means function

unique_pixels=unique(image);
%initialze cluster centers randomly
new=zeros(size(image));
k_means=randsample(unique_pixels,num_k);
tracker=0;
    while tracker<=2
      old_k=k_means;
      for i=1:size(image,1)
            for j=1:size(image,2)
                values=abs(double(k_means)-double(image(i,j)));
                new(i,j)= find(values==min(values),1);
            end
      end
        for i=1:length(k_means)
            matrix=new==i;
            k_means(i)=round(mean(image(matrix)));
        end
        if k_means==old_k
            tracker=tracker+1;
        else
            tracker=0;
        end
    end 
   
end

