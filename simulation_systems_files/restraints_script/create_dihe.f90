program create_dihe
!***************************************************************
!***************************************************************

  implicit none

  integer, parameter :: MAXAT=100000

  integer :: ia,iend,iat,ires,ii,iatindex,ios,iresi,iatlip,ideb
  real(8) :: x,y,z,occ,bfact
  character(len=1) :: typ
  character(len=4) :: rnam!,segnam
  character(len=4) :: atnam
  character(len=5) :: ciat
  character(len=6) :: atty,segnam
  character(len=9) :: cires
  character(len=70), parameter :: fmtin="(a6,a5,1x,a4,1x,a4,a9,3(f8.3),2(f6.2),6x,a6)"
  character(len=70), parameter :: fmtin2="(a6,i5,1x,a4,1x,a4,i5)"
  character(len=70), parameter :: fmtdi="(a,4(1x,a),1x,a)"
  !character(len=70), parameter :: fmtin="(a6,a5,1x,a4,1x,a4,a9,3(f8.3),2(f6.2),6x,a4)"
  character(len=100) :: inam,onam,line

  integer, dimension(1:MAXAT) :: lipiat,lipires
  character(len=4), dimension(1:MAXAT) :: lipatnam,liprnam
  character(len=5) :: cc1,cc3,cc2,co21,cc28,cc29,cc210,cc211,cc13,coc2,cc12,cc11

!---------------------------------- read input
  ia=iargc()
  if (ia /= 1) then
     write(6,"(/2x,a,$)") "Input file (.pdb) : "
!1    format(/2x,"Input file (.pdb) : ",$)
     read(5,"(a)") inam
  else
     call getarg(1,inam)
  endif
!---------------------------------- read input file and get all lipids
  open(1,file=trim(inam)//".pdb",status="old")
  iatlip=0
  do while (.true.)
     read(1,"(a)",iostat=ios) line
     if (ios /= 0) exit
     if (line(1:4) == "ATOM") then
        read(line,fmtin) atty,ciat,atnam,rnam,cires,x,y,z,occ,bfact,segnam
        if (segnam(1:4) == "MEMB") then
           iatlip=iatlip+1
           read(line,fmtin2) atty,lipiat(iatlip),lipatnam(iatlip),liprnam(iatlip),lipires(iatlip)
        endif
     endif
  enddo
  close(1)

!---------------------------------- create dihe.txt fle
  open(11,file="dihe.txt",status="unknown")
  ires=-10
  ii=1
  do while (ii <= iatlip)
     if (lipires(ii) /= ires) then 
        ires=lipires(ii)
        rnam=liprnam(ii)
        ideb=lipiat(ii)-1
        if (rnam == "POPE") then
           write(cc1,"(i5)") ideb+15
           write(cc3,"(i5)") ideb+26
           write(cc2,"(i5)") ideb+18
           write(co21,"(i5)") ideb+20
           write(cc28,"(i5)") ideb+50
           write(cc29,"(i5)") ideb+53
           write(cc210,"(i5)") ideb+55
           write(cc211,"(i5)") ideb+57
           write(11,fmtdi) "IMPROPER",trim(adjustl(cc1)),trim(adjustl(cc3)),trim(adjustl(cc2)),trim(adjustl(co21)),"$FC -120.0"
           write(11,fmtdi) "DIHEDRAL",trim(adjustl(cc28)),trim(adjustl(cc29)),trim(adjustl(cc210)),trim(adjustl(cc211)),"$FC  0.0"
        elseif (rnam == "POPG") then
           write(cc13,"(i5)") ideb
           write(coc2,"(i5)") ideb+7
           write(cc12,"(i5)") ideb+5
           write(cc11,"(i5)") ideb+9
           write(cc1,"(i5)") ideb+17
           write(cc3,"(i5)") ideb+28
           write(cc2,"(i5)") ideb+20
           write(co21,"(i5)") ideb+22
           write(cc28,"(i5)") ideb+52
           write(cc29,"(i5)") ideb+55
           write(cc210,"(i5)") ideb+57
           write(cc211,"(i5)") ideb+59
           write(11,fmtdi) "IMPROPER",trim(adjustl(cc13)),trim(adjustl(coc2)),trim(adjustl(cc12)),trim(adjustl(cc11)),"$FC  120.0"
           write(11,fmtdi) "IMPROPER",trim(adjustl(cc1)),trim(adjustl(cc3)),trim(adjustl(cc2)),trim(adjustl(co21)),"$FC -120.0"
           write(11,fmtdi) "DIHEDRAL",trim(adjustl(cc28)),trim(adjustl(cc29)),trim(adjustl(cc210)),trim(adjustl(cc211)),"$FC  0.0"
        endif
     endif
     ii=ii+1
  enddo
  close(11)


end program create_dihe
