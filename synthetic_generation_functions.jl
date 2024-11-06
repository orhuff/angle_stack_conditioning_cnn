
using Interpolations, FourierTools, Random, Flux, Distributions, NPZ

function NewSrate(ingth, old, new)
   s = size(ingth)
   rlin = (s[1]-1)*old
   nto = round(Int, rlin/new)+1
   out = zeros(Float32, nto, s[2])
   for tr in 1:s[2]
     out[:,tr] = FourierTools.resample(ingth[:,tr],nto)
   end
   return out
end

###########################################################

function Hamming(n::Int)
    n==1 && return [1.]
    n -= 1
    k = collect(0:1:n)
    hamm = 0.54 .- 0.46 .* cos.(2pi.*k/n)
end


############################################################

function Ormsby(; dt=0.004, f=[2.0, 10.0, 40.0, 60.0])

    f1 = f[1]
    f2 = f[2]
    f3 = f[3]
    f4 = f[4]

    fc = (f2+f3)/2.0
    nw = 2.2/(fc*dt)
    nc = floor(Int, nw/2)
    t = dt*collect(-nc:1:nc)
    nw = 2*nc + 1
    a4 = (pi*f4)^2/(pi*(f4-f3))
    a3 = (pi*f3)^2/(pi*(f4-f3))
    a2 = (pi*f2)^2/(pi*(f2-f1))
    a1 = (pi*f1)^2/(pi*(f2-f1))

    u = a4*(sinc.(f4*t)).^2 - a3*(sinc.(f3*t)).^2
    v = a2*(sinc.(f2*t)).^2 - a1*(sinc.(f1*t)).^2

    w = u - v
    w = w.*Hamming(nw)/maximum(w)
end

############################################################

function convWav(in, wav, velocity)
	
	out = zeros(Float32,size(in))
	freqs = collect(fftshift(fftfreq(length(wav), 250)))[ceil(Int8, length(wav)/2):end]

	Q[200:end] = rand(80:140).*ones(Int, 1252)

    @threads for ix = 1:length(in[1,:])
		ts = Flux.unsqueeze(Flux.unsqueeze(in[:,ix],2),2)

		for it = 1:length(in[:,1])
		
			spec = rfft(wav)
			decay = zeros(ceil(Int8, length(wav)/2))
			c = 1
			
			for f in freqs
				decay[c] = exp(-pi * f * sqrt((it*0.004-0.4)^2 + (ix*50)^2/(velocity[it])^2) / (2*Q[it]))
				c = c+1
			end
			
			spec_new = spec .* decay
			wav_new = irfft(spec_new, length(wav))
			
			flt = Flux.unsqueeze(Flux.unsqueeze(wav_new,2),2)
			pad = floor(Int,length(flt)/2)
	
			out[it,ix] = NNlib.conv(ts, flt, pad=pad)[it,1,1]			
		end
	end
	return out
end

############################################################

function SeisNMO(in;dt=0.004,offset=1000.,tnmo=[0.],vnmo=[1500.],max_stretch=1000)

	nt,nx = size(in)
	if length(offset) < size(in,2)
		offset = offset[1]*fill!(similar(in[1,:]),one(eltype(in[1,:])))
	end

	if (length(vnmo) == 1)
		tnmo = convert(Float64,tnmo[1]);
		vnmo = convert(Float64,vnmo[1]);
		ti = collect(0:1:nt-1)*dt
		vi = ones(1,nt)*vnmo

	else
		tnmo = (convert(Array{Float64,1},vec(tnmo)),)
		vnmo = convert(Array{Float64,1},vec(vnmo))
		ti = collect(0:1:nt-1)*dt

		g = interpolate(tnmo, vnmo, Gridded(Linear()))
		ge = extrapolate(g, Line())
		vi = ge(ti)
	end
	out = zeros(typeof(in[1,1]),size(in))
	M = zeros(nt,1)
	@threads for it = 1:nt
		for ix = 1:nx
			time = sqrt(ti[it]^2 + (offset[ix]/vi[it]).^2)
			stretch = (time-ti[it])/(ti[it]+1e-10)
			if (stretch<max_stretch/100)
				its = round(Int,time/dt)+1
				it1 = round(Int,floor(time/dt))+1
				it2 = it1+1
				a = its-it1
				if (it2 <= nt)
					out[it,ix] = (1-a)*in[it1,ix]+a*in[it2,ix]
				end
			end
		end
	end
	return out
end

############################################################

function moveout2nd_aniso(t0, v, x, eta)

   aniso = (2*eta*x^4) / (v^2 * (t0^2*v^2 + (1+2*eta)*x^2))
   t = sqrt(t0^2 + (x^2)/(v^2) - aniso)

end

############################################################

function moveout2nd(t0, v, x)

   t = sqrt(t0^2 + (x^2)/(v^2))
   
end

############################################################

function NMO(gth, v, dt, offs; internal_srate=0.0005)
	nt, nx = size(gth)
	dti = internal_srate
	out = zeros(Float32, size(gth))
	tmp = NewSrate(gth,dt,dti)
	for ix = 1:nx
    	for it = 2:nt
		  tm = moveout2nd((it-1)*dt, v[it], offs[ix])
		  its = floor(Int, tm/dti)+1
		  if its<=size(tmp)[1]
             out[it,ix] = tmp[its, ix]
		  end
	   end
	end
	return out
  end
  
 ############################################################

function getInd(t, dt)
   Int(round(t/dt))
end

############################################################

function NMO_aniso(gth, v, dt, offs, etas; internal_srate=0.0005)
	nt, nx = size(gth)
	dti = internal_srate
	out = zeros(Float32, size(gth))
	tmp = NewSrate(gth,dt,dti)
	
	
	@threads for ix = 1:nx
    	for it = 2:nt
		  
		  tm = moveout2nd_aniso((it-1)*dt, v[it], offs[ix], etas[it,ix])
		  its = floor(Int, tm/dti)+1
		  if its<=size(tmp)[1]
             out[it,ix] = tmp[its, ix]
		  end
	   end
	end
	return out
  end

############################################################

function Vrms(vint, dt)
	vrms = zeros(Float32, length(vint))
	vrms[1] = vint[1]
	vint2 = vint .^ 2
   for i =2:length(vint)
     vrms[i] = sqrt((dt*sum(vint2[1:i]))/(dt*(i-1)))
   end
   return vrms
end

############################################################

function angCalc(t0,off,vi, vr)
   th2 = (off .^ 2) .* (vi .^ 2) ./ ((vr .^ 2) .* ((vr .^ 2) .* (t0 .^ 2) .+ (off .^ 2)))
end

############################################################

function angMatrix(vi, vr, dt, offs)
	rl = length(vi)
    ang2 = zeros(Float32, rl,length(offs))
    @threads for ti = 1:rl
		t0 = (ti-1)*dt
		xi = 0
		for x = offs
			xi += 1
            ang2[ti,xi] = angCalc(t0, x, vi[ti], vr[ti])
		end
	end
    return ang2
end

############################################################

function samplerefl(r)
	dp = Uniform(-r,r)
	ds = Uniform(-r,r)
	rp = rand(dp)
	rp = sign(rp)*log(abs(rp))
	rs = rand(ds)
	rs = sign(rs)*log(abs(rs))
	vpvs=2

	return rp, vpvs*rs-rp
  end

############################################################

### Function synthPrimaries ###
### Generate primaries based on velocities v, record length rl, trace spacing dx and sample rate dt
### Pertubate velocities by random p percent
# TODO wbt, mute

function synthPrimaries(wb, start, ang, vr, p, offs, dt, nstrongs, rp, method::String = "2-term"; vcut=2500.0)
   Random.seed!(abs(rand(Int,1)[1]))


   d = Uniform(-rp,rp)

   noff = length(offs)
   gth = zeros(nt,noff)
   gth_id = zeros(nt,noff)
    
   etas = zeros(nt,noff)
   mean_value = 0 
   std_dev = 0.015 
	
   vp = copy(vr)
   len = length(vp)	
   indices = sort(randperm(len))
   idx = 1
   while idx <= len-20
	   chunk_length = rand(5:20)	
	   if idx + chunk_length < len
	
		   vp[idx:idx+chunk_length-1] *= (1 + (2*rand()-1)*p)
		   etas[idx:idx+chunk_length-1, :] = ones(chunk_length,noff) .* (std_dev * randn(1) .+ mean_value)	
			
	   end
	   idx += chunk_length
   end

	
   vp[idx:end] *= (1 + (2*rand()-1)*p)
	

   vp_id = vr .* (1.0f0)
   
   si = round(Int,start/dt)
   movwb = round.(Int,([moveout2nd((wb+si-1)*dt, vp_id[wb+si], x) for x in offs] .- 1)./dt)
   movwb_id = round.(Int,([moveout2nd((wb+si-1)*dt, vp_id[wb+si], x) for x in offs] .- 1)./dt)
  if method == "2-term" 
  
	if nstrongs > 0
	strong_refls = rand(wb+round(Int,start/dt):1:nt, nstrongs)

	nmlt = copy(strong_refls)
	for i in strong_refls
		num = rand(2:6)
		int_range = -num:num
		for n in int_range
			push!(nmlt, i+n)
		end	
	end
	nmlt = sort(nmlt)
	end


   @threads for ti = wb+round(Int,start/dt):nt

	 mlt = 1
	 if ti == wb || ti == wb+1 || ti == wb+2
		mlt = rand(18:25)
	 end
	 
	 if ti in wb+3:wb+rand(80:120)
		mlt = rand(1:5)
	 end
	 
	 if ti in nmlt
		mlt = rand(1.5:4.2)
     end

	 r0,g = samplerefl(rp)

	 r0_mult = r0*mlt

	 xi = 0
	 gmod = rand(Uniform(1.05,8))
	 g = g*gmod
	 
	 prevt = (ti-1)*dt
     prevo = 0.0
	 for x in offs
		xi = xi + 1
				
		mov_id = moveout2nd((ti-1)*(dt), vp_id[ti], x)

		r = r0 + g*sin(ang[ti,xi])^2
		r = r*mlt
		
		if round(Int,mov_id/dt) <= nt && ((x-prevo)/(mov_id-prevt)) > vcut
		  gth[round(Int,mov_id/dt),xi] += r
		  gth_id[round(Int,mov_id/dt),xi] += r
		end
		prevt = mov_id
        prevo = x
	 end
   end
  end
  return gth, gth_id, vp, vp_id, movwb, movwb_id, etas
end

#######################################################

function angMute!(gth, angmtrx, m)
    nt, nx = size(gth)
    m = sin(deg2rad(m))^2
	for ti = 2:nt
		for xi = 1:nx
           if angmtrx[ti,xi] >= m
              gth[ti,xi:end] .= 0f0
			  break
		   end
		end
	end
end

#########################################################

function angMuteRange!(gth, angmtrx, m, l)
    nt, nx = size(gth)
    m = sin(deg2rad(m))^2
	l = sin(deg2rad(l))^2
	for ti = 2:nt
		for xi = 1:nx
		   if angmtrx[ti,xi] <= l
				gth[ti,1:xi] .= 0f0
		   end		
           if angmtrx[ti,xi] >= m
				gth[ti,xi:end] .= 0f0
		   end

		end
	end
	return gth
end

############################################################

function mute!(gth, m)
   nt,nx = size(gth)
   
   for i = 1:nx
	if m[i] > nt
		m[i] = nt
	end
      gth[1:m[i],i] .= 0f0
   end
end

############################################################

function randWav(dt, fmin1, fmin2, fmax1, fmax2)
	nyq = floor(Int, 1/(2*dt))
    f1 = rand(fmin1:fmin2)
	f2 = f1 + rand(1:8)
    f3 = rand(fmax1:fmax2)
	f4 = min(f3+rand(7:23), nyq)
	wavelet = Ormsby(; dt=dt, f=[f1, f2, f3, f4])
	params = [f1, f2, f3, f4]
	return wavelet, params
end

############################################################

function ruggleGth!(gthin, ruggle)

    s = length(gthin[1,:])
    degrug = rand(1:ruggle)
	rug = sin.(0.75*range(1,degrug;length=s))
	 
	 num = rand(4:20)
	 ind_rand = rand(200:1380, num)
	 @threads for r in ind_rand
		 width = rand(10:50)
		 taper = Int(floor(width/5))
		 indices = range(r, r+width, step=1)
		 taperup = range(r+width, r+width+taper, step=1)
		 taperdown = range(r-taper, r, step=1)
		 randshift = rand(1:10)
		 for i = 1:s
			gthin[indices,i] = shift(gthin[indices,i], 6*rug[i]-randshift)
			gthin[taperup,i] = shift(gthin[taperup,i], 1.5*rug[i]-randshift)
			gthin[taperdown,i] = shift(gthin[taperdown,i], 1.5*rug[i]-randshift)
		 end
	  end
	  
   return gthin
end



#############################################################


function randTraceFromJS3D(io, bs)
    rng = lrange(io)
	vels = zeros(length(rng[1]), bs)
    for i = 1:bs
	   xl = rand(rng[2])
	   il = rand(rng[3])
       vels[:,i] = readtrcs(io, :, xl,xl, il:il)
	end
	return vels
end

moving_average(vs,n) = [sum(@view vs[i:(i+n-1)])/n for i in 1:(length(vs)-(n-1))]

Q = npzread("Q_vals_init.npy")

############################################################

function genDataPairs_fullGathers(bsize, starting, nt, offs, dt, wb, p, v, outsize, angm, ruggle, smoothing; linoise=0, rpp = 0.1, rgp = 0.25, rpm = 0.2, rgm = 0.05, minmov=50.0, max_apexshft = 1, ringnoise = [0,5,5])
	  
	  if length(size(v)) >= 2
		nv = size(v)[2]
	  else
		nv = 1
	  end
 
      nx = length(offs)
	  counting = 0
      @threads for g = 1:bsize
          vi = v[:,rand(1:nv)]	
		  vr = Vrms(vi,dt)
		  ang2 = angMatrix(vi, vr, dt, offs)
		  
		  nstrongs = rand(2:7)
		  rp = 0.2
		  gth, gth_id, vp, vp_id, movwb, movwb_id, etas = synthPrimaries(wb, 0.0, ang2, vr, p, offs, dt, nstrongs, rp, "2-term")
	  
          wav,params = randWav(dt, 2,5,30,0.75/(2*dt))

		  n = smoothing
		  vpsm = cat(moving_average(vp,n), ones(n-1)*vp[end], dims=1)
		  vpsm_id = cat(moving_average(vp_id,n), ones(n-1)*vp_id[end],dims=1)
		  
		  etasm_line = cat(moving_average(etas[:,1],n), ones(n-1)*etas[end,1], dims=1)
		  etasm = repeat(etasm_line, 1, nx)
		  
		  refl_ideal = convWav(gth_id, wav, vr)
		  gthc_id = NMO(refl_ideal, vpsm_id, dt, offs)
		  gthc = NMO_aniso(refl_ideal, vpsm, dt, offs, etasm)

		  if ruggle > 0
			gthc = ruggleGth!(gthc, ruggle)
		  end
		  
		  gthc = angMuteRange!(gthc, ang2, 40, 0)
		  gthc_id = angMuteRange!(gthc_id, ang2, 40, 0)	  
		  
		  scal = 1/maximum(abs.(gthc))

		  npzwrite("./synthetics/angle_matrices/matrix_"*string(g+starting)*".npy", ang2)
		  npzwrite("./synthetics/input_full/inp_"*string(g+starting)*".npy", gthc*scal)
	      npzwrite("./synthetics/target_full/targ_"*string(g+starting)*".npy", gthc_id*scal)
		  counting += 1
		  println("count = " * string(counting) * ", ID = " * string(g)) 
		  end
end		  
