def load(fname):
    # Loads a PLY file with the Stanford bunny mesh.
    # Expects a list of vertices and faces (triangles only) based on the original zipper files.

    file = open(fname)

    # Find the amount of vertices

    STATE_GET_VC = "GET_VC" # Get vertex count
    STATE_GET_FC = "GET_FC" # Get face count
    STATE_FND_HD = "FND_HD" # Find end of header
    STATE_GET_VX = "GET_VX" # Get vertices
    STATE_GET_FS = "GET_FS" # Get faces
    STATE_DONE   = "DONE"   # All vertices and faces found

    vc = 0
    fc = 0
    vx = []
    fs = []
    state = STATE_GET_VC
    for line in file.readlines():
        if (state == STATE_GET_VC):
            if line.startswith("element vertex"):
                (_,_,vcs) = line.split(" ")
                vc = int(vcs)
                state = STATE_GET_FC

        elif (state == STATE_GET_FC):
            if line.startswith("element face"):
                (_,_,fcs) = line.split(" ")
                fc = int(fcs)
                state = STATE_FND_HD

        elif (state == STATE_FND_HD):
             if line.startswith("end_header"):
                state = STATE_GET_VX

        elif (state == STATE_GET_VX):
                (xs,ys,zs) = line.split(" ")[:3]
                vx.append((float(xs), float(ys), float(zs)))
                vc = vc - 1
                if (vc == 0):
                     state = STATE_GET_FS

        elif (state == STATE_GET_FS):
                (i1s, i2s, i3s) = line.split(" ")[:3]
                fs.append((int(i1s), int(i2s), int(i3s)))
                fc = fc - 1
                if (fc == 0):
                     state = STATE_DONE

    if (state != STATE_DONE):
            print("Error, stuck in", state)

    return vx, fs

if (__name__ == "__main__"):
     # Quick test
     vx, fs = load("bun_zipper_res4.ply")
     print("Got", len(vx),"vertices and", len(fs),"triangles")