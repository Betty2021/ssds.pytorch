from __future__ import division
import torch
from math import sqrt as sqrt
from math import ceil as ceil
from itertools import product as product

class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """
    def __init__(self, image_size, feature_maps, aspect_ratios, scale, archor_stride=None, archor_offest=None, clip=True):
        super(PriorBox, self).__init__()
        self.image_size = image_size #[height, width]
        self.feature_maps = feature_maps #[(height, width), ...]
        self.aspect_ratios = aspect_ratios
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(aspect_ratios)
        self.clip = clip
        # scale value
        if isinstance(scale[0], list):
            # get max of the result
            self.scales = [max(s[0] / self.image_size[0], s[1] / self.image_size[1]) for s in scale]
        elif isinstance(scale[0], float) and len(scale) == 2:
            num_layers = len(feature_maps)
            min_scale, max_scale = scale
            self.scales = [min_scale + (max_scale - min_scale) * i / (num_layers - 1) for i in range(num_layers)] + [1.0]
        else: #[0.025,0.08, 0.16, 0.32, 0.6]
            self.scales = scale   
        
        #if archor_stride:
        #    self.steps = [(steps[0] / self.image_size[0], steps[1] / self.image_size[1]) for steps in archor_stride]
        #else:
        if False:
            print("<<<<<<<<<<auto steps>>>>>>>>>>>>>>>>>>")
            self.steps = [(1/f_h, 1/f_w) for f_h, f_w in feature_maps]
            print(self.steps)
            print("<<<<<<<<<<auto steps>>>>>>>>>>>>>>>>>>")

            if archor_offest:
                self.offset = [[offset[0] / self.image_size[0], offset[1] * self.image_size[1]] for offset in archor_offest]
            else:
                self.offset = [[steps[0] * 0.5, steps[1] * 0.5] for steps in self.steps]

        else:
            #self.steps = [(1/f_h, 1/f_w) for f_h, f_w in feature_maps[0:1] ] + \
            #             [(2/f_h, 2/f_w) for f_h, f_w in feature_maps[0:-1] ]
            num_feature_layers= len(feature_maps)
            self.steps = [(16*(2**i)/image_size[0], 16*(2**i)/image_size[1]) for i in range(num_feature_layers) ]
            self.offset = [[steps[0] * 0.5, steps[1] * 0.5] for steps in self.steps]

            #for f_h0, f_w0 in self.feature_maps[0:-1], self.feature_maps[1:])):
            #    self.steps.append( (2.0/f_h0,  2.0/f_w0))

            #self.steps = [(1/f_h, 1/f_w) for f_h, f_w in feature_maps]

    def get_anchor_number(aspect_ratios ):
       anchor_number_list=[]
       for k, aspect_ratio in enumerate(aspect_ratios):
            num_anch =1
            for ar in aspect_ratio:
                ar_sqrt = sqrt(ar)
                anchor_w = ar_sqrt
                if ar > 0.333: # 0.6:1.8
                    num_anch+=1
                else:
                    num_anch+= ceil(1.0/anchor_w)

            anchor_number_list.append(num_anch)

       return anchor_number_list


    def forward(self):
        mean = []
        aspect=self.image_size[1]/self.image_size[0] # w/h
        #aspect=1.0
        # l = 0
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f[0]), range(f[1])):
                cx = j * self.steps[k][1] + self.offset[k][1]
                cy = i * self.steps[k][0] + self.offset[k][0]
                s_k = self.scales[k]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    ar_sqrt = sqrt(ar)
                    anchor_w = s_k*ar_sqrt
                    if ar > 0.333:
                        mean += [cx, cy, anchor_w, s_k*aspect/ar_sqrt]
                    else:
                        x1=cx-s_k*0.5
                        x2=cx+s_k*0.5
                        while x1 + anchor_w <=x2:
                            mean += [x1+anchor_w*0.5, cy, anchor_w, s_k*aspect/ar_sqrt]
                            x1 = x1+anchor_w

                        if x1 < x2 and x1+ anchor_w >x2:
                            mean += [x2-anchor_w*0.5, cy, anchor_w, s_k*aspect/ar_sqrt]

                # s_k_prime = sqrt(s_k * self.scales[k + 1])
                # for ar in self.aspect_ratios[k]:
                #     ar_sqrt = sqrt(ar)
                #     archor_w = s_k_prime*ar_sqrt
                #     if ar > 0.333:
                #         mean += [cx, cy, archor_w, s_k_prime*aspect/ar_sqrt]
                #     else:
                #         x1=cx-s_k_prime*0.5
                #         x2=cx+s_k_prime*0.5
                #         while x1 + archor_w <=x2:
                #             mean += [x1+archor_w*0.5, cy, archor_w, s_k_prime*aspect/ar_sqrt]
                #             x1 = x1+archor_w
                #
                #         if x1 < x2 and x1+ archor_w >x2:
                #             mean += [x2-archor_w*0.5, cy, archor_w, s_k_prime*aspect/ar_sqrt]


                s_k_prime = sqrt(s_k * self.scales[k + 1])
                mean += [cx, cy, s_k_prime, s_k_prime*aspect]

                    # if isinstance(ar, int):
                    #     if ar == 1:
                    #         # aspect_ratio: 1 Min size
                    #         mean += [cx, cy, s_k, s_k]
                    #
                    #         # aspect_ratio: 1 Max size
                    #         # rel size: sqrt(s_k * s_(k+1))
                    #         s_k_prime = sqrt(s_k * self.scales[k+1])
                    #         mean += [cx, cy, s_k_prime, s_k_prime]
                    #     else:
                    #         ar_sqrt = sqrt(ar)
                    #         mean += [cx, cy, s_k*ar_sqrt, s_k/ar_sqrt]
                    #         mean += [cx, cy, s_k/ar_sqrt, s_k*ar_sqrt]
                    # elif isinstance(ar, list):
                    #     mean += [cx, cy, s_k*ar[0], s_k*ar[1]]
        #     print(f, self.aspect_ratios[k])
        # assert False
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output
