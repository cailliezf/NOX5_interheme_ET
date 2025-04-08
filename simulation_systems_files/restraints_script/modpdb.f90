program modpdb
!***************************************************************
!***************************************************************

  implicit none
  integer :: ia,ios
  real(8) :: x,y,z,occ,bfact,zlim
  character(len=1) :: typ
  character(len=4) :: rnam!,segnam
  character(len=4) :: atnam
  character(len=5) :: ciat
  character(len=6) :: atty,segnam
  character(len=9) :: cires
  character(len=70), parameter :: fmtin="(a6,a5,1x,a4,1x,a4,a9,3(f8.3),2(f6.2),6x,a)"
  character(len=100) :: inam,line,czlim
!---------------------------------- read input
  ia=iargc()
  if (ia /= 2) then
     write(6,"(/2x,a,$)") "Input file (.pdb) : "
     read(5,"(a)") inam
     write(6,"(/2x,a,$)") "zlim : "
     read(5,*) zlim
  else
     call getarg(1,inam)
     call getarg(2,czlim)
     read(czlim,*) zlim
  endif
!---------------------------------- read input file and write output file
  open(1,file=trim(inam)//".pdb",status="old")
  open(11,file="protein.ref",status="unknown")
  open(12,file="popg_head_lower.ref",status="unknown")
  open(13,file="popg_head_upper.ref",status="unknown")
  open(14,file="pope_head_lower.ref",status="unknown")
  open(15,file="pope_head_upper.ref",status="unknown")
  do while (.true.)
     read(1,"(a)",iostat=ios) line
     if (ios /= 0) exit
     if ((line(1:3) == "REM").or.(line(1:3) == "END").or.(line(1:3) == "TER")) then
        write(11,"(a)") trim(line)
        write(12,"(a)") trim(line)
        write(13,"(a)") trim(line)
        write(14,"(a)") trim(line)
        write(15,"(a)") trim(line)
     elseif (line(1:4) == "ATOM") then
        read(line,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,bfact,segnam
        if (segnam(1:4) == "PROA") then
           write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           if ((atnam == " N  ").or. (atnam == " CA ").or.(atnam == " C  ").or.(atnam == " O  ")) then
              write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,1.,trim(segnam)
           else
              typ=adjustl(atnam)
              if (typ(1:1) /= "H") then
                 write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.5,trim(segnam)
              else
                 write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
              endif
           endif
        elseif ((segnam(1:4) == "HETA").or.(segnam(1:4) == "HETB")) then
           write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           typ=adjustl(atnam)
           if (typ(1:1) /= "H") then
              write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.5,trim(segnam)
           else
              write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           endif
        elseif (segnam(1:4) == "MEMB") then
           write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           if (rnam == "POPE") then
              write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
              write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
              if ((atnam == " N  ").or. (atnam == " C12").or.(atnam == " C11").or.(atnam == " P  ").or. &
                  (atnam == " O13").or. (atnam == " O14").or.(atnam == " O11").or.(atnam == " O12")) then
                 if (z < -zlim) then
                    write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,1.,trim(segnam)
                    write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                 elseif (z > zlim) then
                    write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                    write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,1.,trim(segnam)
                 else
                    print *,"Head of ",rnam," ",cires," lipid has a strange z = ",z
                    print *,"Maybe something is wrong or zlim needs to be changed"
                    stop
                 endif
              else
                 write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                 write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)             
              endif
           elseif (rnam == "POPG") then
              write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
              write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
              if ((atnam == " C13").or. (atnam == " OC3").or.(atnam == " C12").or.(atnam == " OC2").or. &
                  (atnam == " C11").or. (atnam == " P  ").or.(atnam == " O13").or.(atnam == " O14").or. &
                  (atnam == " O12").or. (atnam == " O11")) then
                 if (z < -zlim) then
                    write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,1.,trim(segnam)
                    write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                 elseif (z > zlim) then
                    write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                    write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,1.,trim(segnam)
                 else
                    print *,"Head of ",rnam," ",cires," lipid has a strange z = ",z
                    print *,"Maybe something is wrong or zlim needs to be changed"
                    stop
                 endif
              else
                 write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
                 write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)             
              endif
           endif
        else
           write(11,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(12,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(13,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(14,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
           write(15,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,0.,trim(segnam)
        endif
     endif
  enddo
  close(1)
  close(11)
  close(12)
  close(13)
  close(14)
  close(15)

end program modpdb
